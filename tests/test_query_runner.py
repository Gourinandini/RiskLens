"""
Unit Tests — QueryRunner

Tests SQL query validation (security), SELECT execution contract,
schema setup, and error handling for the SQLite backend.
"""

import unittest
from src.talk_to_data.query_runner import QueryRunner


class TestQueryRunnerValidation(unittest.TestCase):
    """Tests for validate_sql() — ensures the security allowlist is enforced."""

    @classmethod
    def setUpClass(cls):
        cls.runner = QueryRunner()

    # ── Allowed statements ────────────────────────────────────────────────────

    def test_simple_select_allowed(self):
        self.assertTrue(self.runner.validate_sql("SELECT * FROM applications LIMIT 10"))

    def test_select_with_where_allowed(self):
        self.assertTrue(
            self.runner.validate_sql("SELECT AVG(AMT_INCOME_TOTAL) FROM applications WHERE TARGET = 1;")
        )

    def test_select_with_group_by_allowed(self):
        self.assertTrue(
            self.runner.validate_sql(
                "SELECT NAME_EDUCATION_TYPE, AVG(TARGET)*100 AS DR FROM applications GROUP BY NAME_EDUCATION_TYPE;"
            )
        )

    def test_select_with_order_by_and_limit_allowed(self):
        self.assertTrue(
            self.runner.validate_sql(
                "SELECT SK_ID_CURR, AMT_CREDIT FROM applications ORDER BY AMT_CREDIT DESC LIMIT 10;"
            )
        )

    def test_select_count_allowed(self):
        self.assertTrue(
            self.runner.validate_sql("SELECT COUNT(*) FROM applications WHERE CODE_GENDER = 'F';")
        )

    # ── Blocked destructive statements ────────────────────────────────────────

    def test_drop_table_blocked(self):
        self.assertFalse(self.runner.validate_sql("DROP TABLE applications;"))

    def test_update_blocked(self):
        self.assertFalse(self.runner.validate_sql("UPDATE applications SET TARGET = 0;"))

    def test_delete_blocked(self):
        self.assertFalse(self.runner.validate_sql("DELETE FROM applications WHERE TARGET = 1;"))

    def test_insert_blocked(self):
        self.assertFalse(
            self.runner.validate_sql("INSERT INTO applications (SK_ID_CURR) VALUES (999);")
        )

    def test_create_blocked(self):
        self.assertFalse(self.runner.validate_sql("CREATE TABLE tmp AS SELECT * FROM applications;"))

    def test_alter_blocked(self):
        self.assertFalse(self.runner.validate_sql("ALTER TABLE applications ADD COLUMN foo TEXT;"))

    def test_truncate_blocked(self):
        self.assertFalse(self.runner.validate_sql("TRUNCATE applications;"))

    def test_exec_blocked(self):
        self.assertFalse(self.runner.validate_sql("EXEC xp_cmdshell('rm -rf /');"))

    def test_sql_comment_injection_blocked(self):
        self.assertFalse(
            self.runner.validate_sql("SELECT * FROM applications -- WHERE TARGET = 0")
        )

    def test_multiple_statements_blocked(self):
        self.assertFalse(
            self.runner.validate_sql("SELECT * FROM applications; DROP TABLE applications;")
        )

    def test_non_select_prefix_blocked(self):
        self.assertFalse(self.runner.validate_sql("SHOW TABLES;"))

    def test_empty_string_blocked(self):
        self.assertFalse(self.runner.validate_sql(""))

    def test_whitespace_only_blocked(self):
        self.assertFalse(self.runner.validate_sql("   "))


class TestQueryRunnerExecution(unittest.TestCase):
    """Tests for execute() — result shape and error handling."""

    @classmethod
    def setUpClass(cls):
        cls.runner = QueryRunner()

    def test_basic_select_returns_rows(self):
        res = self.runner.execute("SELECT SK_ID_CURR, TARGET, RISK_BAND FROM applications LIMIT 5")
        self.assertIsNone(res["error"])
        self.assertEqual(len(res["columns"]), 3)
        self.assertEqual(res["row_count"], 5)
        self.assertEqual(len(res["rows"]), 5)

    def test_columns_in_result(self):
        res = self.runner.execute(
            "SELECT TARGET, RISK_BAND FROM applications LIMIT 1"
        )
        self.assertIn("TARGET", res["columns"])
        self.assertIn("RISK_BAND", res["columns"])

    def test_count_query_returns_single_value(self):
        res = self.runner.execute("SELECT COUNT(*) FROM applications")
        self.assertIsNone(res["error"])
        self.assertEqual(len(res["rows"]), 1)
        self.assertIsInstance(res["rows"][0][0], int)
        self.assertGreater(res["rows"][0][0], 0)

    def test_aggregation_returns_numeric(self):
        res = self.runner.execute(
            "SELECT AVG(AMT_INCOME_TOTAL) AS AVG_INCOME FROM applications WHERE TARGET = 1"
        )
        self.assertIsNone(res["error"])
        self.assertEqual(len(res["rows"]), 1)
        avg = res["rows"][0][0]
        self.assertIsInstance(avg, float)
        self.assertGreater(avg, 0)

    def test_group_by_query_multiple_rows(self):
        res = self.runner.execute(
            "SELECT NAME_EDUCATION_TYPE, COUNT(*) FROM applications GROUP BY NAME_EDUCATION_TYPE"
        )
        self.assertIsNone(res["error"])
        self.assertGreater(res["row_count"], 1)

    def test_unsafe_query_blocked_returns_error(self):
        res = self.runner.execute("DROP TABLE applications;")
        self.assertEqual(res["error"], "unsafe query blocked")
        self.assertEqual(res["rows"], [])
        self.assertEqual(res["row_count"], 0)

    def test_invalid_sql_syntax_returns_error(self):
        res = self.runner.execute("SELECT FROM WHERE;")
        self.assertIsNotNone(res["error"])

    def test_nonexistent_column_returns_error(self):
        res = self.runner.execute(
            "SELECT FAKE_COLUMN_THAT_DOES_NOT_EXIST FROM applications LIMIT 1"
        )
        self.assertIsNotNone(res["error"])

    def test_max_100_rows_enforced(self):
        res = self.runner.execute("SELECT SK_ID_CURR FROM applications")
        self.assertLessEqual(res["row_count"], 100)

    def test_risk_band_values_valid(self):
        res = self.runner.execute(
            "SELECT DISTINCT RISK_BAND FROM applications"
        )
        self.assertIsNone(res["error"])
        band_values = {row[0] for row in res["rows"]}
        self.assertTrue(band_values.issubset({"Low", "Medium", "High"}))

    def test_schema_string_contains_key_columns(self):
        schema = self.runner.get_schema_string()
        self.assertIn("applications", schema)
        self.assertIn("SK_ID_CURR", schema)
        self.assertIn("TARGET", schema)
        self.assertIn("RISK_BAND", schema)
        self.assertIn("EXT_SOURCE_2", schema)


if __name__ == "__main__":
    unittest.main()
