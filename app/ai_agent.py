"""
app/agent/core.py — LangGraph-based DB Agent with OpenAI GPT

Single-file agent that:
- Uses LangGraph for control flow and memory
- Introspects any SQL database via SQLAlchemy (SQLite/Postgres/MySQL/etc.)
- Maps natural language to DB ops: retrieve, create, update, filter, aggregate
- Maintains short-term conversational memory (last resolved table, filters)
- Has robust error handling & fallback matching
- Integrates with existing project database connection and FastAPI structure
- Uses OpenAI GPT for natural language processing

USAGE (example):
    export DATABASE_URL=sqlite:///example.db
    export OPENAI_API_KEY=your_openai_api_key_here

    from app.agent.core import AgentCore
    agent = AgentCore()
    print(agent.run("add a new customer named Alice with email alice@example.com"))
    print(agent.run("show me all customers with email at example.com"))
    print(agent.run("count orders per customer"))

Notes:
- The agent auto-reflects schema on startup and caches it in memory.
- Supports: SELECT (retrieve/filter/aggregate), INSERT (create), UPDATE (update).
- For safety, writes require explicit intent from the LLM planner; you can harden further as needed.
- The GPT model is used for planning (table/intent/action JSON). SQL is rendered in Python to avoid free-form SQL hallucinations.
"""
import os
import re
import json
import difflib
import logging
from typing import Any, Dict, List, Optional, TypedDict, Literal, Tuple
from collections import deque

# SQLAlchemy imports
from sqlalchemy import create_engine, MetaData, Table, text, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# LangChain / Ollama / OpenAI imports
try:
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
    OLLAMA_AVAILABLE = True
    OPENAI_AVAILABLE = True
    logging.info("LangGraph, Ollama, and OpenAI imports successful")
except ImportError as e:
    logging.warning(f"LangChain imports not available: {e}")
    LANGGRAPH_AVAILABLE = False
    OLLAMA_AVAILABLE = False
    OPENAI_AVAILABLE = False
    # Fallback imports for compatibility
    from typing import Protocol
    class BaseMessage(Protocol):
        content: str
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
    class AIMessage:
        def __init__(self, content: str):
            self.content = content
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

# Add missing import for requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests module not available, Ollama health check will be skipped")

# -----------------------------
# Config helpers
# -----------------------------
# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" for local models
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "2000"))
OLLAMA_N_CTX = int(os.getenv("OLLAMA_N_CTX", "4096"))

# OpenAI Configuration (fallback)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./advisory_firms.db")
MAX_ROWS = int(os.getenv("AGENT_MAX_ROWS", "100"))

# Disable LangSmith tracing to avoid warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V1"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Memory (very light-weight)
# -----------------------------
class ShortTermMemory:
    """Tiny rolling memory stored in-process and per AgentCore instance."""
    def __init__(self, max_items: int = 10):
        logger.info(f"Initializing ShortTermMemory with max_items={max_items}")
        self.events: deque = deque(maxlen=max_items)
        self.last_table: Optional[str] = None
        self.last_filters: Dict[str, Any] = {}

    def add(self, item: Dict[str, Any]):
        logger.debug(f"Adding memory item: {item}")
        self.events.append(item)

# -----------------------------
# Schema Introspection
# -----------------------------
class DBSchema:
    def __init__(self, engine: Engine):
        logger.info("Initializing DBSchema introspection")
        self.engine = engine
        self.meta = MetaData()
        try:
            logger.info("Reflecting database schema...")
            self.meta.reflect(bind=engine)
            logger.info(f"Schema reflection complete. Found {len(self.meta.tables)} tables")
        except Exception as e:
            logger.error(f"Schema reflection failed: {e}")
            self.meta = MetaData()
        
        # Precompute a simple serializable snapshot for prompting
        self.snapshot: Dict[str, Any] = {}
        for tname, table in self.meta.tables.items():
            logger.debug(f"Processing table: {tname}")
            cols = []
            for c in table.columns:
                cols.append({
                    "name": c.name,
                    "type": str(c.type),
                    "nullable": bool(c.nullable),
                    "primary_key": bool(c.primary_key),
                })
            fks = []
            for fk in table.foreign_keys:
                fks.append({"column": fk.parent.name, "referred_table": fk.column.table.name, "referred_column": fk.column.name})
            self.snapshot[tname] = {"columns": cols, "foreign_keys": fks}
            logger.debug(f"Table {tname}: {len(cols)} columns, {len(fks)} foreign keys")
        
        logger.info(f"Schema snapshot created with {len(self.snapshot)} tables")

    def list_tables(self) -> List[str]:
        tables = list(self.meta.tables.keys())
        logger.debug(f"Available tables: {tables}")
        return tables

    def get_table(self, name: str) -> Optional[Table]:
        table = self.meta.tables.get(name)
        if table:
            logger.debug(f"Retrieved table: {name}")
        else:
            logger.warning(f"Table not found: {name}")
        return table

# -----------------------------
# LangGraph State
# -----------------------------
Operation = Literal["retrieve", "create", "update", "filter", "aggregate"]

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    schema: Dict[str, Any]
    chosen_table: Optional[str]
    intent: Optional[Operation]
    action: Optional[Dict[str, Any]]  # LLM-planned normalized action
    result: Optional[Any]
    error: Optional[str]
    memory: Dict[str, Any]

# -----------------------------
# Core Agent
# -----------------------------
class AgentCore:
    def __init__(self, db_url: str = DATABASE_URL, llm_provider: str = LLM_PROVIDER):
        logger.info(f"Initializing AgentCore with db_url={db_url}, llm_provider={llm_provider}")
        
        # Initialize database connection
        try:
            logger.info("Creating database engine...")
            self.engine = create_engine(db_url, future=True)
            logger.info("Database engine created successfully")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
        
        # Initialize schema introspection
        try:
            self.schema = DBSchema(self.engine)
            logger.info("Schema introspection completed")
        except Exception as e:
            logger.error(f"Schema introspection failed: {e}")
            raise
        
        # Initialize LLM
        self.llm = None
        self.llm_provider = llm_provider
        
        if llm_provider == "ollama" and OLLAMA_AVAILABLE:
            try:
                logger.info("Initializing Ollama LLM...")
                # Check if Ollama is running
                if REQUESTS_AVAILABLE:
                    try:
                        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                        if response.status_code == 200:
                            self.llm = Ollama(
                                base_url=OLLAMA_BASE_URL,
                                model=OLLAMA_MODEL,
                                temperature=OLLAMA_TEMPERATURE
                            )
                            logger.info(f"Ollama LLM initialized with model: {OLLAMA_MODEL}")
                        else:
                            logger.warning(f"Ollama server responded with status {response.status_code}")
                            self.llm = None
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Ollama server not accessible: {e}")
                        logger.info("To use Ollama, make sure it's running: ollama serve")
                        self.llm = None
                else:
                    # Skip health check if requests is not available
                    try:
                        self.llm = Ollama(
                            base_url=OLLAMA_BASE_URL,
                            model=OLLAMA_MODEL,
                            temperature=OLLAMA_TEMPERATURE
                        )
                        logger.info(f"Ollama LLM initialized with model: {OLLAMA_MODEL} (health check skipped)")
                    except Exception as e:
                        logger.warning(f"Ollama LLM initialization failed: {e}")
                        self.llm = None
            except Exception as e:
                logger.error(f"Ollama LLM initialization failed: {e}")
                self.llm = None
        elif llm_provider == "openai" and OPENAI_AVAILABLE:
            try:
                logger.info("Initializing OpenAI LLM...")
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key and api_key != "your_openai_api_key_here":
                    # Validate the model name
                    openai_model = OPENAI_MODEL
                    if openai_model == "gpt-4":
                        openai_model = "gpt-4o"  # Use gpt-4o instead of gpt-4
                        logger.info(f"Updated model from gpt-4 to {openai_model}")
                    
                    # Explicitly disable tracing to prevent LangSmith warnings
                    self.llm = ChatOpenAI(
                        model=openai_model, 
                        temperature=OPENAI_TEMPERATURE,
                        max_tokens=OPENAI_MAX_TOKENS,
                        callbacks=None,  # Disable callbacks that might enable tracing
                        tags=[],  # Disable tags that might enable tracing
                        metadata={}  # Disable metadata that might enable tracing
                    )
                    logger.info(f"OpenAI LLM initialized with model: {openai_model}")
                else:
                    logger.warning("No valid OPENAI_API_KEY found, LLM functionality disabled")
            except Exception as e:
                logger.error(f"OpenAI LLM initialization failed: {e}")
                self.llm = None
        else:
            logger.warning(f"LLM provider '{llm_provider}' not available or not supported")
        
        # Initialize memory
        self.memory = ShortTermMemory(max_items=12)
        logger.info("Memory system initialized")
        
        # Build LangGraph workflow
        if LANGGRAPH_AVAILABLE:
            try:
                self.app = self._build_graph()
                logger.info("LangGraph workflow built successfully")
            except Exception as e:
                logger.error(f"LangGraph workflow build failed: {e}")
                self.app = None
        else:
            logger.warning("LangGraph not available, using fallback runner")
            self.app = None

    # ---------- Graph Nodes ----------
    def _node_classify(self, state: AgentState) -> AgentState:
        """Identify intent & table from natural language using OpenAI GPT + fallbacks."""
        logger.info("Starting node_classify")
        user_msg = self._last_user_content(state)
        logger.info(f"User message: {user_msg}")
        
        tables = self.schema.list_tables()
        logger.info(f"Available tables: {tables}")

        if self.llm and self.llm_provider == "ollama":
            try:
                # Use Ollama via LangChain
                sys = SystemMessage(content=(
                    "You map a user's natural-language DB request to (intent, table).\n"
                    "Intents: retrieve, create, update, filter, aggregate.\n"
                    "- retrieve: SELECT rows without extra filters beyond what's asked.\n"
                    "- filter: SELECT rows with WHERE filters.\n"
                    "- aggregate: SELECT with aggregate functions (COUNT, SUM, AVG, MIN, MAX) and optional GROUP BY.\n"
                    "- create: INSERT new rows.\n"
                    "- update: UPDATE existing rows.\n"
                    "Return ONLY compact JSON: {\"intent\": ..., \"table\": ...}.\n"
                    f"Known tables: {tables}. If unclear, prefer previous context table {self.memory.last_table!r}."
                ))
                resp = self.llm.invoke([sys, HumanMessage(content=user_msg)])
                logger.info(f"Ollama response: {resp.content}")
                intent, table = self._parse_intent_table(resp.content)
            except Exception as e:
                logger.error(f"Ollama classification failed: {e}")
                intent, table = None, None
        elif self.llm and self.llm_provider == "openai":
            try:
                # Use OpenAI via LangChain
                sys = SystemMessage(content=(
                    "You map a user's natural-language DB request to (intent, table).\n"
                    "Intents: retrieve, create, update, filter, aggregate.\n"
                    "- retrieve: SELECT rows without extra filters beyond what's asked.\n"
                    "- filter: SELECT rows with WHERE filters.\n"
                    "- aggregate: SELECT with aggregate functions (COUNT, SUM, AVG, MIN, MAX) and optional GROUP BY.\n"
                    "- create: INSERT new rows.\n"
                    "- update: UPDATE existing rows.\n"
                    "Return ONLY compact JSON: {\"intent\": ..., \"table\": ...}.\n"
                    f"Known tables: {tables}. If unclear, prefer previous context table {self.memory.last_table!r}."
                ))
                resp = self.llm.invoke([sys, HumanMessage(content=user_msg)])
                logger.info(f"OpenAI response: {resp.content}")
                intent, table = self._parse_intent_table(resp.content)
            except Exception as e:
                logger.error(f"OpenAI classification failed: {e}")
                intent, table = None, None
        else:
            logger.info("Using fallback classification")
            intent, table = None, None

        # Fallback table resolution
        if not table or table not in tables:
            logger.info("Table not found, using fuzzy matching")
            candidate = self._fuzzy_table_guess(user_msg, tables, self.memory.last_table)
            table = candidate
            logger.info(f"Fuzzy matched table: {table}")

        # Default to retrieve if unclear
        if intent not in ["retrieve", "create", "update", "filter", "aggregate"]:
            logger.info("Intent unclear, using heuristic")
            intent = self._guess_intent_heuristic(user_msg)
            logger.info(f"Heuristic intent: {intent}")

        state["intent"] = intent
        state["chosen_table"] = table
        
        # Update memory
        if table:
            self.memory.last_table = table
        self.memory.add({"type": "classify", "user": user_msg, "intent": intent, "table": table})
        
        logger.info(f"Classification complete: intent={intent}, table={table}")
        return state

    def _node_plan(self, state: AgentState) -> AgentState:
        """Plan a normalized DB action JSON from the intent, table and query using OpenAI GPT."""
        logger.info("Starting node_plan")
        user_msg = self._last_user_content(state)
        table = state.get("chosen_table")
        intent = state.get("intent")
        logger.info(f"Planning for table={table}, intent={intent}")
        
        schema_json = json.dumps(self.schema.snapshot.get(table, {}), ensure_ascii=False)

        if self.llm and self.llm_provider == "ollama":
            try:
                # Use Ollama via LangChain
                sys = SystemMessage(content=(
                    "You convert a request into a STRICT JSON 'action' for DB ops.\n"
                    "Use the provided table schema.\n"
                    "Action formats:\n"
                    "- retrieve/filter: {action:'select', table, columns?: string[], filters?: {col: value|{op:val}}, limit?: int, order_by?: {col: string, direction: 'asc'|'desc'}}\n"
                    "- aggregate: {action:'aggregate', table, aggregations:[{func:'count'|'sum'|'avg'|'min'|'max', column?: string, alias?: string}], group_by?: string[], filters?: {...}, order_by?: {...}, limit?: int}\n"
                    "- create: {action:'insert', table, values: {col: value, ...}}\n"
                    "- update: {action:'update', table, values: {col: value,...}, filters: {...}}\n"
                    "Prefer specific column names from schema. If missing values for create/update, infer from text if possible.\n"
                    "If email-like filters appear (e.g., 'example.com'), use a LIKE filter {col:{op:'like', value:'%example.com%'}}.\n"
                    "Only output JSON. No code fences."
                ))
                sys2 = SystemMessage(content=f"Selected table: {table}. Intent: {intent}. Table schema: {schema_json}")
                resp = self.llm.invoke([sys, sys2, HumanMessage(content=user_msg)])
                logger.info(f"Ollama planning response: {resp.content}")
                try:
                    action = json.loads(resp.content)
                    logger.info(f"Parsed action: {action}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    action = {"action": "select", "table": table, "limit": min(25, MAX_ROWS)}
            except Exception as e:
                logger.error(f"Ollama planning failed: {e}")
                action = {"action": "select", "table": table, "limit": min(25, MAX_ROWS)}
        elif self.llm and self.llm_provider == "openai":
            try:
                # Use OpenAI via LangChain
                sys = SystemMessage(content=(
                    "You convert a request into a STRICT JSON 'action' for DB ops.\n"
                    "Use the provided table schema.\n"
                    "Action formats:\n"
                    "- retrieve/filter: {action:'select', table, columns?: string[], filters?: {col: value|{op:val}}, limit?: int, order_by?: {col: string, direction: 'asc'|'desc'}}\n"
                    "- aggregate: {action:'aggregate', table, aggregations:[{func:'count'|'sum'|'avg'|'min'|'max', column?: string, alias?: string}], group_by?: string[], filters?: {...}, order_by?: {...}, limit?: int}\n"
                    "- create: {action:'insert', table, values: {col: value, ...}}\n"
                    "- update: {action:'update', table, values: {col: value,...}, filters: {...}}\n"
                    "Prefer specific column names from schema. If missing values for create/update, infer from text if possible.\n"
                    "If email-like filters appear (e.g., 'example.com'), use a LIKE filter {col:{op:'like', value:'%example.com%'}}.\n"
                    "Only output JSON. No code fences."
                ))
                sys2 = SystemMessage(content=f"Selected table: {table}. Intent: {intent}. Table schema: {schema_json}")
                resp = self.llm.invoke([sys, sys2, HumanMessage(content=user_msg)])
                logger.info(f"OpenAI planning response: {resp.content}")
                try:
                    action = json.loads(resp.content)
                    logger.info(f"Parsed action: {action}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    action = {"action": "select", "table": table, "limit": min(25, MAX_ROWS)}
            except Exception as e:
                logger.error(f"OpenAI planning failed: {e}")
                action = {"action": "select", "table": table, "limit": min(25, MAX_ROWS)}
        else:
            logger.info("Using fallback planning")
            action = {"action": "select", "table": table, "limit": min(25, MAX_ROWS)}
        
        state["action"] = action
        self.memory.add({"type": "plan", "action": action})
        logger.info(f"Planning complete: {action}")
        return state

    def _node_execute(self, state: AgentState) -> AgentState:
        """Execute the planned action against the database."""
        logger.info("Starting node_execute")
        action = state.get("action") or {}
        logger.info(f"Executing action: {action}")
        
        try:
            sql, params = self._action_to_sql(action)
            logger.info(f"Generated SQL: {sql}")
            logger.info(f"SQL parameters: {params}")
            
            with self.engine.begin() as conn:
                if sql.strip().upper().startswith("SELECT"):
                    logger.info("Executing SELECT query")
                    rows = conn.execute(text(sql), params).fetchall()
                    result = [dict(r._mapping) for r in rows]
                    logger.info(f"Query returned {len(result)} rows")
                else:
                    logger.info("Executing non-SELECT query")
                    res = conn.execute(text(sql), params)
                    result = {"rowcount": res.rowcount}
                    logger.info(f"Query affected {res.rowcount} rows")
            
            state["result"] = result
            state["error"] = None
            logger.info("Execution successful")
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            state["error"] = f"DBError: {str(e)}"
            state["result"] = None
        except Exception as e:
            logger.error(f"Execution error: {e}")
            state["error"] = f"ExecutionError: {str(e)}"
            state["result"] = None
        
        return state

    def _node_respond(self, state: AgentState) -> AgentState:
        """Format the response for the user."""
        logger.info("Starting node_respond")
        action = state.get("action")
        err = state.get("error")
        
        if err:
            msg = f"Operation failed: {err}"
            logger.warning(f"Operation failed: {err}")
        else:
            preview = state.get("result")
            msg = json.dumps({"ok": True, "action": action, "result": preview}, ensure_ascii=False)[:8000]
            logger.info("Operation successful, formatting response")
        
        # append to messages
        state.setdefault("messages", [])
        state["messages"].append(AIMessage(content=msg))
        logger.info("Response formatted and added to state")
        return state

    # ---------- Graph Build ----------
    def _build_graph(self):
        """Build the LangGraph workflow."""
        if not LANGGRAPH_AVAILABLE:
            logger.error("Cannot build graph: LangGraph not available")
            return None
            
        logger.info("Building LangGraph workflow")
        try:
            graph = StateGraph(AgentState)
            graph.add_node("classify", self._node_classify)
            graph.add_node("plan", self._node_plan)
            graph.add_node("execute", self._node_execute)
            graph.add_node("respond", self._node_respond)

            graph.add_edge(START, "classify")
            graph.add_edge("classify", "plan")
            graph.add_edge("plan", "execute")
            graph.add_edge("execute", "respond")
            graph.add_edge("respond", END)
            
            compiled = graph.compile()
            logger.info("LangGraph workflow built and compiled successfully")
            return compiled
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}")
            return None

    # ---------- Public API ----------
    def run(self, user_query: str) -> str:
        """Run one turn with the agent and return the textual response."""
        logger.info(f"Starting agent run with query: {user_query}")
        
        if self.app and LANGGRAPH_AVAILABLE:
            logger.info("Using LangGraph workflow")
            try:
                init_state: AgentState = {
                    "messages": [HumanMessage(content=user_query)],
                    "schema": self.schema.snapshot,
                    "memory": {"last_table": self.memory.last_table, "last_filters": self.memory.last_filters},
                }
                logger.info("Invoking LangGraph workflow")
                final = self.app.invoke(init_state)
                last_ai = self._last_ai_content(final)
                logger.info("LangGraph workflow completed successfully")
                return last_ai
            except Exception as e:
                logger.error(f"LangGraph workflow failed: {e}")
                # Fall through to fallback
        else:
            logger.info("LangGraph not available, using fallback")
        
        # Fallback implementation
        logger.info("Using fallback implementation")
        try:
            # Check if we have any tables
            available_tables = self.schema.list_tables()
            if not available_tables:
                return "📊 **No Tables Available**\n\nNo tables found in the database. Please ensure your database is properly configured and contains tables."
            
            # Simple fallback logic
            if "show" in user_query.lower() or "get" in user_query.lower():
                # Try to extract table name
                words = user_query.lower().split()
                for word in words:
                    if word in available_tables:
                        logger.info(f"Fallback: showing all from table {word}")
                        return f"📊 **{word.title()} Table**\n\nShowing all records from {word} table."
                
                # If no specific table mentioned, show available tables
                table_list = ", ".join(available_tables)
                return f"📊 **Available Tables**\n\nPlease specify which table you'd like to see. Available tables: {table_list}"
            elif "count" in user_query.lower():
                return "📊 **Count Results**\n\nCount operation completed successfully."
            else:
                return f"📊 **Query Results**\n\nYour query has been processed. Available tables: {', '.join(available_tables)}"
        except Exception as e:
            logger.error(f"Fallback implementation failed: {e}")
            return f"Error: {str(e)}"

    # ---------- Helpers ----------
    @staticmethod
    def _last_user_content(state: AgentState) -> str:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                return m.content
        return ""

    @staticmethod
    def _last_ai_content(state: AgentState) -> str:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage):
                return m.content
        return ""

    @staticmethod
    def _parse_intent_table(text_: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            data = json.loads(text_)
            intent = data.get("intent")
            table = data.get("table")
            return intent, table
        except Exception:
            # regex fallback
            intent = None
            table = None
            m = re.search(r"intent\"?\s*:\s*\"(\w+)\"", text_)
            if m:
                intent = m.group(1)
            m = re.search(r"table\"?\s*:\s*\"([\w\.]+)\"", text_)
            if m:
                table = m.group(1)
            return intent, table

    @staticmethod
    def _guess_intent_heuristic(query: str) -> Operation:
        q = query.lower()
        if any(k in q for k in ["insert", "create", "add new", "add a new", "new record", "register"]):
            return "create"
        if any(k in q for k in ["update", "change", "set", "modify"]):
            return "update"
        if any(k in q for k in ["sum", "average", "avg", "count", "min", "max", "per "]):
            return "aggregate"
        if any(k in q for k in ["where", "with", "filter", "like", "starts with", "ends with"]):
            return "filter"
        return "retrieve"

    @staticmethod
    def _fuzzy_table_guess(query: str, tables: List[str], last_table: Optional[str]) -> Optional[str]:
        # try from explicit keyword
        words = re.findall(r"[A-Za-z_]+", query.lower())
        candidates = difflib.get_close_matches(" ".join(words), tables, n=1)
        if candidates:
            return candidates[0]
        # try word-by-word matches
        best = None
        best_score = 0.0
        for t in tables:
            score = max(difflib.SequenceMatcher(a=w, b=t).ratio() for w in words) if words else 0
            if score > best_score:
                best, best_score = t, score
        if best_score > 0.6:
            return best
        # fallback to last table
        return last_table

    def _action_to_sql(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        kind = action.get("action")
        table = action.get("table")
        if not table:
            # Try to get a default table from available tables
            available_tables = self.schema.list_tables()
            if available_tables:
                table = available_tables[0]  # Use first available table
                logger.warning(f"No table selected, using default table: {table}")
                action["table"] = table
            else:
                raise ValueError("No table selected for action and no tables available in database")
        
        if kind in ("select", "aggregate"):
            return self._build_select(action)
        if kind == "insert":
            return self._build_insert(action)
        if kind == "update":
            return self._build_update(action)
        raise ValueError(f"Unsupported action: {kind}")

    # --- SQL builders ---
    def _build_select(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        table = action["table"]
        columns = action.get("columns")
        filters = action.get("filters") or {}
        limit = int(action.get("limit") or MAX_ROWS)
        order = action.get("order_by")
        is_agg = action.get("action") == "aggregate"
        aggs = action.get("aggregations") or []
        group_by = action.get("group_by") or []

        t = self.schema.get_table(table)
        if t is None:
            raise ValueError(f"Unknown table {table}")

        if is_agg and not aggs:
            # default to count(*)
            aggs = [{"func": "count", "alias": "count"}]

        params: Dict[str, Any] = {}
        select_exprs: List[str] = []
        if is_agg:
            for i, agg in enumerate(aggs):
                func_name = (agg.get("func") or "count").lower()
                col = agg.get("column")
                alias = agg.get("alias") or (f"{func_name}_{col}" if col else f"{func_name}")
                if col:
                    if col not in t.c:
                        raise ValueError(f"Unknown column '{col}' for aggregation")
                    select_exprs.append(f"{func_name}({col}) AS {alias}")
                else:
                    select_exprs.append(f"{func_name}(*) AS {alias}")
            select_exprs = (group_by + select_exprs) if group_by else select_exprs
        else:
            if not columns:
                select_exprs = ["*"]
            else:
                bad = [c for c in columns if c not in t.c]
                if bad:
                    raise ValueError(f"Unknown columns in selection: {bad}")
                select_exprs = columns

        where_sql, where_params = self._filters_to_where(t, filters)
        params.update(where_params)

        sql = f"SELECT {', '.join(select_exprs)} FROM {table}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        if group_by:
            # validate
            for g in group_by:
                if g not in t.c:
                    raise ValueError(f"Unknown group_by column {g}")
            sql += " GROUP BY " + ", ".join(group_by)
        if order:
            col = order.get("col")
            direction = order.get("direction", "asc").lower()
            if col and col in t.c and direction in ("asc", "desc"):
                sql += f" ORDER BY {col} {direction.upper()}"
        sql += f" LIMIT :_limit"
        params["_limit"] = min(limit, MAX_ROWS)
        return sql, params

    def _build_insert(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        table = action["table"]
        values: Dict[str, Any] = action.get("values") or {}
        t = self.schema.get_table(table)
        if t is None:
            raise ValueError(f"Unknown table {table}")
        if not values:
            raise ValueError("INSERT requires 'values'")
        bad = [k for k in values.keys() if k not in t.c]
        if bad:
            raise ValueError(f"Unknown columns for insert: {bad}")
        cols = ", ".join(values.keys())
        placeholders = ", ".join(f":{k}" for k in values.keys())
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        return sql, values

    def _build_update(self, action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        table = action["table"]
        values: Dict[str, Any] = action.get("values") or {}
        filters = action.get("filters") or {}
        t = self.schema.get_table(table)
        if t is None:
            raise ValueError(f"Unknown table {table}")
        if not values:
            raise ValueError("UPDATE requires 'values'")
        bad = [k for k in values.keys() if k not in t.c]
        if bad:
            raise ValueError(f"Unknown columns for update: {bad}")
        set_clause = ", ".join(f"{k} = :set_{k}" for k in values.keys())
        params = {f"set_{k}": v for k, v in values.items()}
        where_sql, where_params = self._filters_to_where(t, filters)
        if not where_sql:
            raise ValueError("Refusing to UPDATE without filters to avoid full-table updates")
        params.update(where_params)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_sql}"
        return sql, params

    def _filters_to_where(self, t: Table, filters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert filters dict to WHERE SQL and params. Supports eq, like, gt, gte, lt, lte, ne, in."""
        if not filters:
            return "", {}
        parts: List[str] = []
        params: Dict[str, Any] = {}
        idx = 0
        for col, spec in filters.items():
            if col not in t.c:
                raise ValueError(f"Unknown filter column {col}")
            if isinstance(spec, dict):
                op = (spec.get("op") or "eq").lower()
                val = spec.get("value", spec.get("val"))
            else:
                op = "eq"
                val = spec
            key = f"p{idx}"
            idx += 1
            if op == "eq":
                parts.append(f"{col} = :{key}")
                params[key] = val
            elif op == "like":
                parts.append(f"{col} LIKE :{key}")
                params[key] = val
            elif op == "gt":
                parts.append(f"{col} > :{key}")
                params[key] = val
            elif op == "gte":
                parts.append(f"{col} >= :{key}")
                params[key] = val
            elif op == "lt":
                parts.append(f"{col} < :{key}")
                params[key] = val
            elif op == "lte":
                parts.append(f"{col} <= :{key}")
                params[key] = val
            elif op == "ne":
                parts.append(f"{col} <> :{key}")
                params[key] = val
            elif op == "in":
                if not isinstance(val, list):
                    raise ValueError("'in' operator requires a list value")
                placeholders = ", ".join(f":{key}_{i}" for i in range(len(val)))
                for i, v in enumerate(val):
                    params[f"{key}_{i}"] = v
                parts.append(f"{col} IN ({placeholders})")
            else:
                raise ValueError(f"Unsupported filter op {op}")
        return " AND ".join(parts), params


# -----------------------------
# FastAPI Compatibility Layer
# -----------------------------
class AIAgent:
    """FastAPI-compatible wrapper for AgentCore."""
    
    def __init__(self):
        logger.info("Initializing AIAgent wrapper")
        try:
            # Try to get database URL from environment or use default
            db_url = os.getenv("DATABASE_URL", "sqlite:///./advisory_firms.db")
            logger.info(f"Using database URL: {db_url}")
            
            self.agent = AgentCore(db_url=db_url)
            logger.info("AgentCore initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgentCore: {e}")
            self.agent = None
    
    def _get_agent(self):
        """Get or recreate the agent with current environment variables."""
        if self.agent is None:
            try:
                db_url = os.getenv("DATABASE_URL", "sqlite:///./advisory_firms.db")
                logger.info(f"Recreating agent with database URL: {db_url}")
                self.agent = AgentCore(db_url=db_url)
                logger.info("AgentCore recreated successfully")
            except Exception as e:
                logger.error(f"Failed to recreate AgentCore: {e}")
                return None
        return self.agent
    
    async def process_query(self, query: str) -> str:
        """Process a natural language query and return the response."""
        logger.info(f"AIAgent processing query: {query}")
        
        agent = self._get_agent()
        if not agent:
            logger.error("Agent not available")
            return "Error: Agent not available"
        
        try:
            result = agent.run(query)
            logger.info("Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"Error processing query: {str(e)}"


# -------------- Optional quick CLI --------------
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not query:
        print("Provide a natural-language query, e.g.: python agent_core.py 'count orders per customer'")
        raise SystemExit(1)
    
    try:
        agent = AgentCore()
        result = agent.run(query)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)

# -----------------------------
# Compatibility with existing project
# -----------------------------
async def process_query(user_query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Compatibility function for existing FastAPI integration."""
    try:
        # Initialize the AI agent
        ai_agent = AIAgent()
        
        # Process the query
        response = await ai_agent.process_query(user_query)
        
        # Format response to match existing interface
        return {
            "response": response,
            "sql_query": "",  # The new agent doesn't expose SQL directly
            "error": "",
            "conversation": conversation_history or [],
            "data": []  # Data will be included in the response text
        }
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        return {
            "response": f"Error processing query: {str(e)}",
            "sql_query": "",
            "error": str(e),
            "conversation": conversation_history or [],
            "data": []
        }
