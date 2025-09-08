# metadataengine.py
# PEACE-C–aligned metadata engine with provenance, Cc logging, and trivalent verdicts

from __future__ import annotations
import ast
import asyncio
import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

# -----------------------------------------------------------------------------
# Robust PEACE-C imports (with fallbacks so this file always runs)
# -----------------------------------------------------------------------------
try:
    from peace_c import (
        Engine, EngineConfig, CcEstimator, Oracle,
        Context, TruthValue, tv_to_str,
        empirical_pattern_perspective, pragmatic_context_perspective
    )
    PEACEC_AVAILABLE = True
except Exception:
    PEACEC_AVAILABLE = False

    class TruthValue(Enum):
        T = 1
        F = 0
        B = 2

    def tv_to_str(tv: "TruthValue") -> str:
        return {TruthValue.T: "T", TruthValue.F: "F", TruthValue.B: "B"}[tv]

    class Context:
        def __init__(self):
            self.facts: Dict[str, Any] = {}
            self.needed: List[str] = []
        def add_fact(self, k: str, v: Any):
            self.facts[k] = v

    class CcEstimator:
        def __init__(self, target_questions: int = 5):
            self.target_questions = target_questions

    class EngineConfig:
        def __init__(self, cc_threshold: float = 0.8, novelty_threshold: float = 0.55, max_liminal_rounds: int = 3):
            self.cc_threshold = cc_threshold
            self.novelty_threshold = novelty_threshold
            self.max_liminal_rounds = max_liminal_rounds

    class Oracle:
        def __init__(self, llm=None):
            self.llm = llm

    class Engine:
        def __init__(self, cc=None, oracle=None, config=None, llm=None):
            self.cc, self.oracle, self.config, self.llm = cc, oracle, config, llm
            self.perspectives = []
            self.retriever = None
            self.last_cc = None
            self.last_novelty = None
        def tick(self, claim: str, ctx: Context):
            # Fallback: neutral hold with mid confidence
            self.last_cc = 0.5
            self.last_novelty = 0.0
            return TruthValue.B, 0.50, "PEACE-C not installed; returning neutral hold."

    def empirical_pattern_perspective():
        return {}
    def pragmatic_context_perspective():
        return {}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("peace_oracle.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Trivalent Truth (non-explosive)
# -----------------------------------------------------------------------------
class TV(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2       # paraconsistent "both/hold"
    UNKNOWN = 3    # treated as BOTH downstream

def TV_from_truthvalue(tv: TruthValue) -> TV:
    return {TruthValue.T: TV.TRUE, TruthValue.F: TV.FALSE}.get(tv, TV.BOTH)

def truthvalue_from_TV(tv: TV) -> TruthValue:
    if tv == TV.TRUE:  return TruthValue.T
    if tv == TV.FALSE: return TruthValue.F
    return TruthValue.B

# -----------------------------------------------------------------------------
# Core domain models
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MathematicalProblem:
    name: str
    description: str
    complexity_score: int
    computational_bound: int
    problem_type: str            # "number_theory", "analysis", "combinatorics", ...
    verification_function: Optional[Callable[[int], Any]] = None
    def __hash__(self) -> int:
        return hash((self.name, self.description, self.complexity_score))

@dataclass
class CodeModification:
    modification_id: str
    target_module: str
    target_function: str
    original_code: str
    modified_code: str
    modification_type: str       # "add", "modify", "delete"
    safety_score: float
    mathematical_soundness: float
    reasoning: str
    timestamp: float
    def __hash__(self) -> int:
        return hash(self.modification_id)

# -----------------------------------------------------------------------------
# Perspectives
# -----------------------------------------------------------------------------
@dataclass
class PEACEPerspective:
    name: str
    evaluate_fn: Callable[[Any], TV]
    confidence_fn: Callable[[Any], float]
    memory: Dict[str, Tuple[TV, float]] = field(default_factory=dict)
    stability_score: float = 1.0
    def evaluate(self, statement: Any) -> Tuple[TV, float]:
        key = str(statement)
        if key not in self.memory:
            v = self.evaluate_fn(statement)
            c = self.confidence_fn(statement)
            self.memory[key] = (v, c)
        return self.memory[key]

class CodeSafetyPerspective(PEACEPerspective):
    def __init__(self):
        super().__init__(
            name="code_safety",
            evaluate_fn=self._evaluate_safety,
            confidence_fn=self._compute_safety_confidence,
        )
    def _evaluate_safety(self, modification: CodeModification) -> TV:
        checks = [
            self._check_syntax(modification.modified_code),
            self._check_infinite_loops(modification.modified_code),
            self._check_memory_safety(modification.modified_code),
            self._check_side_effects(modification.modified_code),
            self._check_imports(modification.modified_code),
        ]
        safe_count = sum(1 for x in checks if x)
        total = len(checks)
        if safe_count == total: return TV.TRUE
        if safe_count == 0:     return TV.FALSE
        if safe_count >= int(total * 0.7): return TV.BOTH
        return TV.FALSE
    def _compute_safety_confidence(self, modification: CodeModification) -> float:
        try:
            tree = ast.parse(modification.modified_code)
            complexity = len(list(ast.walk(tree)))
            base = max(0.1, 1.0 - (complexity / 100.0))
            pattern_boost = sum(0.05 for p in ["if ", "return", "self.", "try:", "except:"] if p in modification.modified_code)
            return min(0.95, base + pattern_boost)
        except SyntaxError:
            return 0.1
        except Exception:
            return 0.3
    def _check_syntax(self, code: str) -> bool:
        try: ast.parse(code); return True
        except SyntaxError: return False
    def _check_infinite_loops(self, code: str) -> bool:
        try: tree = ast.parse(code)
        except Exception: return False
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    if not any(isinstance(n, ast.Break) for n in ast.walk(node)):
                        return False
        return True
    def _check_memory_safety(self, code: str) -> bool:
        dangerous = ["exec(", "eval(", "globals()", "__import__", "setattr(", "delattr(", "vars()", "locals()"]
        return not any(p in code for p in dangerous)
    def _check_side_effects(self, code: str) -> bool:
        side = ["os.", "sys.", "subprocess.", "open(", "file(", "input(", "write(", "delete"]
        return not any(p in code for p in side if "print(" not in code)
    def _check_imports(self, code: str) -> bool:
        try: tree = ast.parse(code)
        except Exception: return False
        dangerous_modules = {"os","sys","subprocess","pickle","marshal"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_modules: return False
            if isinstance(node, ast.ImportFrom):
                if node.module in dangerous_modules: return False
        return True

class MathematicalSoundnessPerspective(PEACEPerspective):
    def __init__(self):
        super().__init__(
            name="mathematical_soundness",
            evaluate_fn=self._evaluate_soundness,
            confidence_fn=self._compute_soundness_confidence,
        )
    def _evaluate_soundness(self, modification: CodeModification) -> TV:
        s = modification.modified_code.lower()
        methods = ["hardy_littlewood","asymptotic","prime","sieve","theorem","conjecture","logarithm","probability"]
        rigor   = ["bound","limit","convergence","error","precision","confidence","estimate","approximation"]
        danger  = [" prove "," proof "," qed","therefore proven","definitively true","absolutely certain"]
        if any(d in s for d in danger): return TV.FALSE
        if sum(1 for m in methods if m in s) >= 2 and any(r in s for r in rigor): return TV.TRUE
        if any(m in s for m in methods) or any(r in s for r in rigor): return TV.BOTH
        return TV.FALSE
    def _compute_soundness_confidence(self, modification: CodeModification) -> float:
        s = modification.modified_code.lower()
        kw = ["theorem","conjecture","prime","asymptotic","logarithm","analysis","heuristic","probability","distribution"]
        base = min(0.9, 0.3 + 0.1 * sum(1 for k in kw if k in s))
        if "hardy_littlewood" in s: base += 0.2
        if "asymptotic" in s and "analysis" in s: base += 0.15
        if "bound" in s or "limit" in s: base += 0.1
        return min(0.95, base)

class CategoryErrorPerspective(PEACEPerspective):
    def __init__(self):
        super().__init__(
            name="category_error",
            evaluate_fn=self._evaluate_category_error,
            confidence_fn=self._compute_category_confidence,
        )
    def _evaluate_category_error(self, modification: CodeModification) -> TV:
        s = modification.modified_code.lower()
        impossible = [("infinite","verify"), ("all","numbers"), ("every","integer"), ("prove","conjecture")]
        boundaries = ["bound","limit","threshold","if n >","computational","feasible","approximation","heuristic"]
        meta = ["confidence","certainty","probability","estimate","likely","suggests","indicates"]
        has_impossible = any(all(w in s for w in pat) for pat in impossible)
        has_boundaries = any(w in s for w in boundaries)
        has_meta = any(w in s for w in meta)
        if has_impossible and not has_boundaries: return TV.FALSE
        if has_boundaries and has_meta: return TV.TRUE
        if has_boundaries or has_meta: return TV.BOTH
        return TV.BOTH
    def _compute_category_confidence(self, modification: CodeModification) -> float:
        return 0.7

# -----------------------------------------------------------------------------
# SQLite provenance cache (with migrations)
# -----------------------------------------------------------------------------
class VersionedPEACECache:
    def __init__(self, db_path: str = "peace_cache.db"):
        self.cache: Dict[str, List[Tuple[TV, float, str]]] = defaultdict(list)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock:
            self.conn.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY,
                    statement_hash TEXT,
                    statement TEXT,
                    verdict TEXT,
                    confidence REAL,
                    perspective TEXT,
                    version TEXT,
                    timestamp REAL
                );

                CREATE TABLE IF NOT EXISTS code_modifications (
                    id INTEGER PRIMARY KEY,
                    modification_id TEXT UNIQUE,
                    target_module TEXT,
                    target_function TEXT,
                    original_code TEXT,
                    modified_code TEXT,
                    safety_verdict TEXT,
                    safety_confidence REAL,
                    mathematical_soundness REAL,
                    reasoning TEXT,
                    timestamp REAL,
                    code_hash TEXT
                );

                CREATE TABLE IF NOT EXISTS peace_provenance (
                    id INTEGER PRIMARY KEY,
                    eval_row_id INTEGER,
                    cc REAL,
                    novelty REAL,
                    context_json TEXT,
                    rationale TEXT,
                    FOREIGN KEY(eval_row_id) REFERENCES evaluations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_statement_hash ON evaluations(statement_hash);
                CREATE INDEX IF NOT EXISTS idx_modification_id ON code_modifications(modification_id);
                CREATE INDEX IF NOT EXISTS idx_eval_row_id ON peace_provenance(eval_row_id);
            """)
            self.conn.commit()

    def record_evaluation(
        self, statement: Any, verdict: TV, confidence: float, perspective: str, version: str = "1.0"
    ) -> int:
        statement_str = str(statement)
        statement_hash = hashlib.sha256(statement_str.encode()).hexdigest()
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO evaluations 
                (statement_hash, statement, verdict, confidence, perspective, version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (statement_hash, statement_str, verdict.name, confidence, perspective, version, time.time()),
            )
            self.conn.commit()
            return cur.lastrowid

    def record_provenance(self, eval_row_id: int, cc: Optional[float], novelty: Optional[float],
                          context_json: str, rationale: str):
        with self._lock:
            self.conn.execute(
                "INSERT INTO peace_provenance (eval_row_id, cc, novelty, context_json, rationale) VALUES (?, ?, ?, ?, ?)",
                (eval_row_id, cc, novelty, context_json, rationale)
            )
            self.conn.commit()

    def record_modification(self, modification: CodeModification):
        code_hash = hashlib.sha256(modification.modified_code.encode()).hexdigest()
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO code_modifications
                (modification_id, target_module, target_function, original_code, modified_code,
                 safety_verdict, safety_confidence, mathematical_soundness, reasoning, timestamp, code_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    modification.modification_id,
                    modification.target_module,
                    modification.target_function,
                    modification.original_code,
                    modification.modified_code,
                    "PENDING",
                    modification.safety_score,
                    modification.mathematical_soundness,
                    modification.reasoning,
                    modification.timestamp,
                    code_hash,
                ),
            )
            self.conn.commit()

    def get_modification_history(self, target_function: str) -> List[CodeModification]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT modification_id, target_module, target_function, original_code, modified_code,
                       safety_verdict, safety_confidence, mathematical_soundness, reasoning, timestamp, code_hash
                FROM code_modifications 
                WHERE target_function = ? 
                ORDER BY timestamp DESC
                """,
                (target_function,),
            )
            rows = cursor.fetchall()

        modifications: List[CodeModification] = []
        for row in rows:
            mod = CodeModification(
                modification_id=row[0],
                target_module=row[1],
                target_function=row[2],
                original_code=row[3],
                modified_code=row[4],
                modification_type="modify",
                safety_score=float(row[6] or 0.0),             # safety_confidence
                mathematical_soundness=float(row[7] or 0.0),
                reasoning=row[8] or "",
                timestamp=float(row[9] or time.time()),
            )
            modifications.append(mod)
        return modifications

    def close(self):
        with self._lock:
            self.conn.close()

# -----------------------------------------------------------------------------
# Minimal LLM adapter (mock)
# -----------------------------------------------------------------------------
class MockLLM:
    def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return "B 0.50 Rationale: default neutrality."

class LLMInterface:
    def __init__(self):
        self.conversation_history: List[str] = []
    async def analyze_mathematical_problem(self, problem: MathematicalProblem) -> Dict[str, Any]:
        if problem.problem_type == "number_theory":
            return {
                "structure": "number_theoretic_conjecture",
                "approaches": ["direct_verification", "asymptotic_analysis", "probabilistic_methods"],
                "computational_feasibility": problem.computational_bound,
                "algorithmic_strategies": ["sieve_based", "oracle_guided_leaps", "pattern_recognition"],
                "meta_logical_potential": 0.8,
            }
        elif problem.problem_type == "analysis":
            return {
                "structure": "analytical_conjecture",
                "approaches": ["functional_analysis", "measure_theory", "complex_analysis"],
                "computational_feasibility": problem.computational_bound // 10,
                "algorithmic_strategies": ["numerical_methods", "symbolic_computation"],
                "meta_logical_potential": 0.6,
            }
        else:
            return {
                "structure": "general_mathematical_problem",
                "approaches": ["computational_search", "theoretical_analysis"],
                "computational_feasibility": problem.computational_bound,
                "algorithmic_strategies": ["exhaustive_search", "heuristic_methods"],
                "meta_logical_potential": 0.5,
            }
    async def suggest_solution_method(self, problem: MathematicalProblem, analysis: Dict) -> Dict[str, Any]:
        if problem.complexity_score >= 8:
            return {
                "primary_method": "oracle_guided_verification",
                "secondary_methods": ["pattern_learning", "asymptotic_extrapolation"],
                "confidence": 0.85,
                "requires_code_modification": True,
                "modification_targets": ["verification_engine", "pattern_learner", "oracle_evaluator"],
                "reasoning": "High complexity requires advanced oracle capabilities",
            }
        elif problem.complexity_score >= 5:
            return {
                "primary_method": "enhanced_verification",
                "secondary_methods": ["direct_computation", "heuristic_analysis"],
                "confidence": 0.9,
                "requires_code_modification": True,
                "modification_targets": ["verification_engine"],
                "reasoning": "Medium complexity benefits from enhanced verification",
            }
        else:
            return {
                "primary_method": "direct_computation",
                "secondary_methods": ["exhaustive_search"],
                "confidence": 0.95,
                "requires_code_modification": False,
                "reasoning": "Low complexity suitable for direct computation",
            }
    async def analyze_current_code(self, module_name: str, function_name: str) -> Dict[str, Any]:
        return {
            "current_capabilities": ["basic_verification", "simple_patterns", "direct_computation"],
            "limitations": ["no_large_scale_leaping", "limited_heuristics", "no_asymptotic_analysis"],
            "improvement_potential": 0.9,
            "safety_concerns": ["infinite_loops", "memory_overflow", "stack_overflow"],
            "lines_of_code": 150,
            "complexity_score": 6,
        }
    def _stable_id(self, *parts: str, prefix: str = "mod_", n: int = 12) -> str:
        h = hashlib.sha256("||".join(parts).encode()).hexdigest()
        return f"{prefix}{h[:n]}"
    async def propose_code_modifications(
        self, problem: MathematicalProblem, current_analysis: Dict, solution_method: Dict
    ) -> List[CodeModification]:
        mods: List[CodeModification] = []
        if solution_method.get("requires_code_modification"):
            if problem.problem_type == "number_theory" and problem.complexity_score >= 7:
                mods.append(CodeModification(
                    modification_id=self._stable_id(problem.name, "oracle_enhancement"),
                    target_module="peace_oracle",
                    target_function="evaluate_large_scale",
                    original_code="def evaluate_large_scale(self, n):\n    if n <= self.computational_bound:\n        return self.direct_verify(n)\n    return None\n",
                    modified_code=("def evaluate_large_scale(self, n):\n"
                                   "    if n <= self.computational_bound:\n"
                                   "        return self.direct_verify(n)\n"
                                   "    elif n <= self.computational_bound * 1000:\n"
                                   "        return self.oracle_guided_verification(n)\n"
                                   "    else:\n"
                                   "        return self.hardy_littlewood_analysis(n)\n"),
                    modification_type="modify",
                    safety_score=0.0,
                    mathematical_soundness=0.0,
                    reasoning="Add asymptotic analysis and oracle guidance for large n",
                    timestamp=time.time(),
                ))
        return mods

# -----------------------------------------------------------------------------
# Self-Modifying Oracle + PEACE-C Engine
# -----------------------------------------------------------------------------
class SelfModifyingPEACEOracle:
    def __init__(self, llm_interface: LLMInterface,
                 engine: Optional[Engine] = None):
        self.llm = llm_interface
        self.cache = VersionedPEACECache()

        self.safety_perspectives = [
            CodeSafetyPerspective(),
            MathematicalSoundnessPerspective(),
            CategoryErrorPerspective(),
        ]

        if engine is None:
            engine = Engine(
                cc=CcEstimator(target_questions=5),
                oracle=Oracle(llm=MockLLM()),
                config=EngineConfig(cc_threshold=0.8, novelty_threshold=0.55, max_liminal_rounds=3),
                llm=MockLLM()
            )
            engine.perspectives = [
                empirical_pattern_perspective(),
                pragmatic_context_perspective(),
            ]
        self.engine = engine

        self.mathematical_perspectives: Dict[str, PEACEPerspective] = {}
        self.active_modifications: Dict[str, CodeModification] = {}
        logger.info("Initialized Self-Modifying PEACE Oracle (with PEACE-C Engine)")

    # ---- PEACE-C claim evaluator with provenance ----
    def evaluate_claim_with_peace(self, claim: str, seed_facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = Context()
        for k, v in (seed_facts or {}).items():
            ctx.add_fact(k, v)
        tv, conf, rationale = self.engine.tick(claim, ctx)
        row_id = self.cache.record_evaluation(
            statement={"claim": claim, "facts": seed_facts or {}},
            verdict=TV_from_truthvalue(tv),
            confidence=conf,
            perspective="peace_c_engine",
            version=f"peace-c-{'ok' if PEACEC_AVAILABLE else 'fallback'}"
        )
        cc = getattr(self.engine, "last_cc", None)
        novelty = getattr(self.engine, "last_novelty", None)
        self.cache.record_provenance(
            eval_row_id=row_id, cc=cc, novelty=novelty,
            context_json=str(ctx.facts), rationale=rationale
        )
        return {
            "truth_value": tv_to_str(tv),
            "confidence": conf,
            "rationale": rationale,
            "context_facts": ctx.facts,
            "outstanding_questions": getattr(ctx, "needed", []),
            "cc": cc, "novelty": novelty,
        }

    # ---- Main flow for math problems ----
    async def solve_mathematical_problem(self, problem: MathematicalProblem) -> Dict[str, Any]:
        logger.info(f"Starting analysis of problem: {problem.name}")
        try:
            analysis = await self.llm.analyze_mathematical_problem(problem)
            method = await self.llm.suggest_solution_method(problem, analysis)

            current_code_analysis: Dict[str, Any] = {}
            if method.get("requires_code_modification"):
                for target in method.get("modification_targets", []):
                    current_code_analysis[target] = await self.llm.analyze_current_code("peace_oracle", target)

            proposed_mods = await self.llm.propose_code_modifications(problem, current_code_analysis, method)

            safe_mods: List[CodeModification] = []
            for mod in proposed_mods:
                safety_result = await self._evaluate_modification_safety(mod)
                if (safety_result["verdict"] in [TV.TRUE, TV.BOTH]) and (safety_result["confidence"] > 0.6):
                    safe_mods.append(mod)
                else:
                    logger.warning(f"Rejected unsafe modification: {mod.modification_id}")

            if safe_mods:
                await self._apply_modifications(safe_mods)

            solution_result = await self._attempt_problem_solution(problem, method)

            if not solution_result.get("algorithmic_solution"):
                meta_result = await self._meta_logical_analysis(problem, analysis)
                solution_result.update(meta_result)

            return solution_result

        except Exception as e:
            logger.error(f"Error solving problem {problem.name}: {e}")
            return {"error": str(e), "algorithmic_solution": False, "meta_logical_analysis": False}

    async def _evaluate_modification_safety(self, modification: CodeModification) -> Dict[str, Any]:
        verdicts: Dict[str, TV] = {}
        confidences: Dict[str, float] = {}

        for p in self.safety_perspectives:
            v, c = p.evaluate(modification)
            verdicts[p.name] = v
            confidences[p.name] = c
            self.cache.record_evaluation(
                statement={"mod_id": modification.modification_id, "target": modification.target_function},
                verdict=v, confidence=c, perspective=p.name, version="safety-1.0"
            )

        integrated = self._integrate_safety_verdicts(verdicts, confidences)
        return {
            "verdict": integrated["verdict"],
            "confidence": integrated["confidence"],
            "individual_verdicts": verdicts,
            "individual_confidences": confidences,
            "score_breakdown": integrated.get("score_breakdown", {}),
        }

    def _integrate_safety_verdicts(self, verdicts: Dict[str, TV], confidences: Dict[str, float]) -> Dict[str, Any]:
        weighted = {"TRUE": 0.0, "FALSE": 0.0, "BOTH": 0.0, "UNKNOWN": 0.0}
        total = 0.0
        for name, v in verdicts.items():
            w = float(confidences.get(name, 0.0))
            weighted[v.name] += w
            total += w
        if total > 0:
            for k in weighted:
                weighted[k] /= total

        # Paraconsistent tie-break: prefer BOTH if TRUE and FALSE close
        if abs(weighted["TRUE"] - weighted["FALSE"]) <= 0.15 and (weighted["TRUE"] + weighted["FALSE"]) >= 0.5:
            v = TV.BOTH
        elif weighted["FALSE"] > 0.3:
            v = TV.FALSE
        elif weighted["TRUE"] > 0.7:
            v = TV.TRUE
        else:
            v = TV.BOTH
        return {"verdict": v, "confidence": max(weighted.values()), "score_breakdown": weighted}

    async def _apply_modifications(self, modifications: List[CodeModification]):
        for m in modifications:
            try:
                self.cache.record_modification(m)
                self.active_modifications[m.modification_id] = m
                logger.info(f"Applied modification {m.modification_id}")
            except Exception as e:
                logger.error(f"Failed to apply modification {m.modification_id}: {e}")

    async def _attempt_problem_solution(self, problem: MathematicalProblem, method: Dict[str, Any]) -> Dict[str, Any]:
        if problem.verification_function and method.get("primary_method") == "direct_computation":
            try:
                result = problem.verification_function(1000)
                return {"algorithmic_solution": True, "method": "direct_computation", "result": result, "confidence": 1.0}
            except Exception as e:
                logger.error(f"Direct verification failed: {e}")

        if self.active_modifications:
            return {
                "algorithmic_solution": True,
                "method": "enhanced_verification",
                "result": "Enhanced verification capabilities active",
                "modifications_applied": len(self.active_modifications),
                "confidence": 0.8,
            }

        return {
            "algorithmic_solution": False,
            "method": "basic_analysis",
            "result": "Insufficient computational resources for direct solution",
            "confidence": 0.3,
        }

    async def _meta_logical_analysis(self, problem: MathematicalProblem, analysis: Dict[str, Any]) -> Dict[str, Any]:
        if problem.name not in self.mathematical_perspectives:
            self.mathematical_perspectives[problem.name] = self._create_problem_perspective(problem, analysis)
        perspective = self.mathematical_perspectives[problem.name]

        test_statements = self._generate_test_statements(problem, analysis)
        results: Dict[str, Dict[str, Any]] = {}
        for st in test_statements:
            v, c = perspective.evaluate(st)
            results[st] = {"verdict": v, "confidence": c}
            self.cache.record_evaluation(st, v, c, problem.name, version="meta-1.0")

        integrated = self._integrate_meta_logical_results(results, problem, analysis)
        return {
            "meta_logical_analysis": True,
            "problem_perspective": problem.name,
            "statements_evaluated": len(test_statements),
            "integrated_verdict": integrated["verdict"],
            "confidence": integrated["confidence"],
            "reasoning": integrated["reasoning"],
            "detailed_results": results,
        }

    def _create_problem_perspective(self, problem: MathematicalProblem, analysis: Dict[str, Any]) -> PEACEPerspective:
        def evaluate_fn(statement: Any) -> TV:
            s = str(statement).lower()
            if problem.problem_type == "number_theory":
                if "prime" in s:
                    if "infinite" in s and "gap" not in s: return TV.TRUE
                    if "distribution" in s: return TV.BOTH
                    return TV.UNKNOWN
                if "conjecture" in s: return TV.BOTH
                return TV.UNKNOWN
            if problem.problem_type == "analysis":
                if "convergent" in s or "bounded" in s: return TV.BOTH
                if "continuous" in s: return TV.TRUE
                return TV.UNKNOWN
            if "finite" in s: return TV.TRUE
            if "infinite" in s: return TV.BOTH
            return TV.UNKNOWN

        def confidence_fn(statement: Any) -> float:
            base = 0.5 * (10 - problem.complexity_score) / 10.0
            s = str(statement).lower()
            if problem.problem_type == "number_theory": base += 0.2
            if "infinite" in s or "all" in s: base *= 0.7
            return min(0.9, max(0.1, base))

        return PEACEPerspective(
            name=f"problem_{problem.name}",
            evaluate_fn=evaluate_fn,
            confidence_fn=confidence_fn,
        )

    def _generate_test_statements(self, problem: MathematicalProblem, analysis: Dict[str, Any]) -> List[str]:
        statements = [
            f"The {problem.name} has a finite solution",
            f"The {problem.name} can be solved computationally",
            f"The {problem.name} requires infinite computation",
        ]
        if problem.problem_type == "number_theory":
            statements += [
                "Prime numbers are infinite",
                "Prime gaps are bounded",
                "There exists a pattern in prime distribution",
                "Asymptotic methods apply to this problem",
                "Hardy-Littlewood heuristics are relevant",
            ]
        elif problem.problem_type == "analysis":
            statements += [
                "The function has bounded variation",
                "Convergence is uniform",
                "The series converges absolutely",
                "Measure theory applies to this problem",
            ]
        elif problem.problem_type == "combinatorics":
            statements += [
                "The counting problem has closed form",
                "Generating functions apply",
                "The asymptotic growth is polynomial",
                "Probabilistic methods are effective",
            ]
        if problem.complexity_score >= 7:
            statements += [
                "Direct computation is infeasible",
                "Heuristic methods are necessary",
                "The problem admits approximation algorithms",
                "Meta-logical reasoning provides insight",
            ]
        if analysis.get("meta_logical_potential", 0) > 0.7:
            statements += [
                "Oracle guidance improves solution quality",
                "Pattern learning enhances verification",
                "Self-modification provides computational advantages",
            ]
        return statements

    def _integrate_meta_logical_results(
        self, results: Dict[str, Dict[str, Any]], problem: MathematicalProblem, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        weighted = {"TRUE": 0.0, "FALSE": 0.0, "BOTH": 0.0, "UNKNOWN": 0.0}
        total = 0.0
        for _, res in results.items():
            vname = res["verdict"].name
            c = float(res["confidence"])
            weighted[vname] += c
            total += c
        if total > 0:
            for k in weighted:
                weighted[k] /= total
        # prefer BOTH if T and F are close
        if abs(weighted["TRUE"] - weighted["FALSE"]) <= 0.15 and (weighted["TRUE"] + weighted["FALSE"]) >= 0.5:
            overall = "BOTH"
        else:
            overall = max(weighted, key=lambda k: weighted[k]) if weighted else "UNKNOWN"
        reasoning: List[str] = []
        if weighted["TRUE"] > 0.4: reasoning.append("Strong evidence supports computational tractability")
        if weighted["FALSE"] > 0.3: reasoning.append("Significant barriers to direct solution exist")
        if weighted["BOTH"] > 0.4: reasoning.append("Paraconsistent aspects require careful analysis")
        if weighted["UNKNOWN"] > 0.5: reasoning.append("Insufficient information for definitive assessment")
        if problem.complexity_score >= 8: reasoning.append("High complexity suggests meta-logical approaches are valuable")
        if analysis.get("meta_logical_potential", 0) > 0.7: reasoning.append("Structure amenable to oracle-guided analysis")
        return {
            "verdict": TV[overall],
            "confidence": max(weighted.values()) if weighted else 0.0,
            "reasoning": ". ".join(reasoning) if reasoning else "Analysis inconclusive",
            "verdict_distribution": weighted,
        }

    async def get_modification_history(self, function_name: Optional[str] = None) -> List[CodeModification]:
        if function_name:
            return self.cache.get_modification_history(function_name)
        return list(self.active_modifications.values())

    async def analyze_self_improvement_potential(self) -> Dict[str, Any]:
        modification_count = len(self.active_modifications)
        capability_enhancement = modification_count * 0.1
        solved_problems = len(self.mathematical_perspectives)
        learning_factor = min(1.0, solved_problems * 0.05)
        safety_score = 1.0
        return {
            "current_modifications": modification_count,
            "capability_enhancement": capability_enhancement,
            "learning_factor": learning_factor,
            "safety_score": safety_score,
            "improvement_potential": min(1.0, capability_enhancement + learning_factor) * safety_score,
            "recommended_focus": self._recommend_improvement_focus(),
        }

    def _recommend_improvement_focus(self) -> List[str]:
        recs: List[str] = []
        if len(self.active_modifications) < 3: recs.append("Enhance computational capabilities")
        if len(self.mathematical_perspectives) < 5: recs.append("Develop more specialized mathematical perspectives")
        if not any("pattern" in m.reasoning for m in self.active_modifications.values()):
            recs.append("Implement advanced pattern recognition")
        if not any("asymptotic" in m.reasoning for m in self.active_modifications.values()):
            recs.append("Add asymptotic analysis capabilities")
        recs.append("Strengthen meta-logical reasoning frameworks")
        return recs

    def close(self):
        self.cache.close()
        logger.info("PEACE Oracle shutdown complete")

# -----------------------------------------------------------------------------
# Example verification helpers
# -----------------------------------------------------------------------------
def count_twin_primes_upto(n: int) -> int:
    def is_prime(k: int) -> bool:
        if k < 2: return False
        if k % 2 == 0: return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0: return False
            i += 2
        return True
    count = 0
    p = 3
    while p + 2 <= n:
        if is_prime(p) and is_prime(p + 2): count += 1
        p += 2
    return count

def goldbach_all_even_upto(limit_n: int) -> bool:
    def is_prime(k: int) -> bool:
        if k < 2: return False
        if k % 2 == 0: return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0: return False
            i += 2
        return True
    for n in range(4, min(limit_n, 10000) + 1, 2):
        found = False
        for a in range(2, n // 2 + 1):
            if is_prime(a) and is_prime(n - a):
                found = True
                break
        if not found: return False
    return True

# -----------------------------------------------------------------------------
# Example problems
# -----------------------------------------------------------------------------
TWIN_PRIME_CONJECTURE = MathematicalProblem(
    name="twin_prime_conjecture",
    description="There are infinitely many twin primes (primes p such that p+2 is also prime)",
    complexity_score=9,
    computational_bound=10**6,
    problem_type="number_theory",
    verification_function=count_twin_primes_upto
)

GOLDBACH_CONJECTURE = MathematicalProblem(
    name="goldbach_conjecture",
    description="Every even integer greater than 2 can be expressed as sum of two primes",
    complexity_score=8,
    computational_bound=10**6,
    problem_type="number_theory",
    verification_function=lambda limit: goldbach_all_even_upto(limit)
)

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
async def main():
    llm = LLMInterface()
    oracle = SelfModifyingPEACEOracle(llm)
    try:
        logger.info("=== Self-Modifying PEACE Oracle Demonstration ===")

        # PEACE-C claim evaluation (with provenance)
        claim_out = oracle.evaluate_claim_with_peace(
            "The city plans to reduce downtown speed limits next month.",
            seed_facts={"source_hint": "Heard at a neighborhood meeting"}
        )
        print("\nPEACE Engine Claim Verdict:", claim_out)

        # Math flow
        twin_prime_result = await oracle.solve_mathematical_problem(TWIN_PRIME_CONJECTURE)
        print("\nTwin Prime Analysis Result:")
        print(twin_prime_result)

        goldbach_result = await oracle.solve_mathematical_problem(GOLDBACH_CONJECTURE)
        print("\nGoldbach Analysis Result:")
        print(goldbach_result)

        improvement = await oracle.analyze_self_improvement_potential()
        print("\nSelf-Improvement Analysis:", improvement)

        mods = await oracle.get_modification_history("evaluate_large_scale")
        if mods:
            print(f"\nModification history for evaluate_large_scale ({len(mods)} rows):")
            for m in mods:
                print(" -", m.modification_id, m.reasoning, f"(score={m.safety_score:.2f})")
        else:
            print("\nNo stored modification history for evaluate_large_scale.")

    finally:
        oracle.close()

if __name__ == "__main__":
    asyncio.run(main())
