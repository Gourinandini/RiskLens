"""
NL-to-SQL Chatbot Agent Module.

Defines the TalkToDataAgent class which orchestrates user query parsing, LLM-based SQL
generation, execution safety validation, SQLite execution, and LLM-based business summary generation.
"""

import pandas as pd
from typing import Dict, Any
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from src.utils.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from src.utils.logger import get_logger
from src.talk_to_data.query_runner import QueryRunner
from src.talk_to_data.prompt_templates import (
    SYSTEM_PROMPT,
    NL_TO_SQL_TEMPLATE,
    ANSWER_TEMPLATE,
)

logger = get_logger(__name__)

class TalkToDataAgent:
    """
    TalkToDataAgent translates natural language questions into safe SQL SELECT queries,
    runs them against the SQLite database, and generates business-oriented summaries.
    """

    def __init__(self) -> None:
        """Initialize LLM, database runner, and conversation memory."""
        logger.info("Initializing TalkToDataAgent...")
        try:
            # Initialize ChatGroq
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE
            )
            
            # Initialize query executor
            self.query_runner = QueryRunner()
            
            # Memory window of k=5 turns
            self.memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="history",
                input_key="input",
                output_key="output"
            )
            logger.info("TalkToDataAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing TalkToDataAgent: {str(e)}")
            raise e

    def ask(self, user_question: str) -> Dict[str, Any]:
        """
        Processes a natural language question. Generates SQL, executes it, and summaries findings.

        Args:
            user_question (str): User question about dataset.

        Returns:
            Dict[str, Any]: Query metadata, SQL output, table results, and plain answer.
        """
        logger.info(f"Received user question: {user_question}")
        
        # Generate SQL query using LLM based on schema and history
        history_vars = self.memory.load_memory_variables({})
        history = history_vars.get("history", "")
        
        user_content = NL_TO_SQL_TEMPLATE.format(history=history, question=user_question)
        
        try:
            messages = [
                ("system", SYSTEM_PROMPT),
                ("human", user_content)
            ]
            response = self.llm.invoke(messages)
            sql_query = response.content.strip()
            
            # Clean markdown code blocks from response
            if sql_query.startswith("```"):
                lines = sql_query.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                sql_query = "\n".join(lines).strip()
                
            logger.info(f"Generated SQL: {sql_query}")
            
        except Exception as e:
            logger.error(f"LLM execution error during SQL generation: {str(e)}")
            return self._error_response(user_question, "", f"LLM error: {str(e)}")

        # Check for query error response
        if "QUERY_ERROR" in sql_query:
            ans = "I'm sorry, I couldn't answer that query based on the applications dataset schema."
            return self._error_response(user_question, sql_query, "QUERY_ERROR", answer=ans)

        # Validate SQL query format
        if not self.query_runner.validate_sql(sql_query):
            ans = "I cannot execute this request. The query contains unauthorized operations (non-SELECT or destructive commands)."
            return self._error_response(user_question, sql_query, "SQL validation blocked", answer=ans)

        # Execute SQL query
        db_result = self.query_runner.execute(sql_query)
        if db_result.get("error"):
            ans = f"There was an error executing the query on the database: {db_result['error']}"
            return self._error_response(user_question, sql_query, db_result["error"], answer=ans)

        # Summarize query output into plain text
        columns = db_result["columns"]
        rows = db_result["rows"]
        
        df_res = pd.DataFrame(rows, columns=columns)
        table_markdown = df_res.to_markdown(index=False) if not df_res.empty else "No rows returned."
        
        answer_prompt = ANSWER_TEMPLATE.format(
            question=user_question,
            sql_query=sql_query,
            result_table=table_markdown
        )
        
        try:
            summary_messages = [
                ("human", answer_prompt)
            ]
            summary_resp = self.llm.invoke(summary_messages)
            plain_answer = summary_resp.content.strip()
            
            # Save transaction context to memory
            self.memory.save_context({"input": user_question}, {"output": plain_answer})
            
            # Return formatted response
            return {
                "question": user_question,
                "sql_query": sql_query,
                "result_table": {
                    "columns": columns,
                    "rows": rows,
                    "row_count": db_result["row_count"]
                },
                "answer": plain_answer,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating text summary: {str(e)}")
            return self._error_response(user_question, sql_query, f"LLM Summary Error: {str(e)}")

    def clear_memory(self) -> None:
        """Reset the conversation memory window."""
        self.memory.clear()
        logger.info("Conversation memory reset.")

    def _error_response(self, question: str, sql: str, error: str, answer: str = None) -> Dict[str, Any]:
        """Generate structured error dictionary response."""
        return {
            "question": question,
            "sql_query": sql,
            "result_table": {"columns": [], "rows": [], "row_count": 0},
            "answer": answer or f"An error occurred while answering your question: {error}",
            "success": False,
            "error": error
        }
