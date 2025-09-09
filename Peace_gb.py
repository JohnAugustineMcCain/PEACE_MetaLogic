#!/usr/bin/env python3
"""
PEACE-Enhanced Goldbach Conjecture Analysis Tool
A comprehensive analyzer using paraconsistent logic and meta-reasoning
for investigating the Goldbach conjecture with heuristic awareness.

Combines:
- Trivalent truth values (T, F, B) for handling uncertainty
- Context completeness tracking (Cc)
- Multiple verification perspectives
- Hardy-Littlewood heuristic integration
- Self-adaptive analysis capabilities
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GoldbachPEACE")

# ==============================================================================
# Core PEACE Framework Components
# ==============================================================================

class TruthValue(Enum):
    """Trivalent truth values for paraconsistent logic"""
    TRUE = auto()
    FALSE = auto() 
    BOTH = auto()  # Paraconsistent "both true and false"

@dataclass
class Evidence:
    """Evidence accumulator with confidence tracking"""
    positive: float = 0.0
    negative: float = 0.0
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    
    def strength(self) -> float:
        """Overall evidence strength"""
        return abs(self.positive - self.negative)
    
    def verdict(self, threshold: float = 0.6) -> TruthValue:
        """Convert evidence to truth value"""
        if self.positive >= threshold and self.negative < threshold:
            return TruthValue.TRUE
        elif self.negative >= threshold and self.positive < threshold:
            return TruthValue.FALSE
        else:
            return TruthValue.BOTH

@dataclass
class AnalysisContext:
    """Context for Goldbach analysis with completeness tracking"""
    number: int
    evidence: Evidence = field(default_factory=Evidence)
    decompositions: List[Tuple[int, int]] = field(default_factory=list)
    heuristic_data: Dict[str, float] = field(default_factory=dict)
    computational_bounds: Dict[str, int] = field(default_factory=dict)
    perspective_verdicts: Dict[str, TruthValue] = field(default_factory=dict)
    questions_answered: int = 0
    questions_total: int = 5
    
    def completeness(self) -> float:
        """Cc: Context completeness measure"""
        if self.questions_total == 0:
            return 1.0
        return min(1.0, self.questions_answered / self.questions_total)
    
    def add_evidence(self, positive: float = 0.0, negative: float = 0.0, 
                    source: str = "unknown"):
        """Add evidence from a source"""
        self.evidence.positive = min(1.0, self.evidence.positive + positive)
        self.evidence.negative = min(1.0, self.evidence.negative + negative)
        self.evidence.sources.append(source)
        self.questions_answered += 1

# ==============================================================================
# Prime Number Utilities
# ==============================================================================

def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Efficient prime sieve"""
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]

class PrimalityTester:
    """Advanced primality testing with multiple algorithms"""
    
    def __init__(self):
        self.prime_cache: Set[int] = set()
        self.composite_cache: Set[int] = set()
        self.small_primes = set(sieve_of_eratosthenes(10000))
    
    def is_prime(self, n: int) -> bool:
        """Determine if n is prime using cached results and Miller-Rabin"""
        if n in self.prime_cache:
            return True
        if n in self.composite_cache:
            return False
        
        if n < 2:
            return False
        if n in self.small_primes:
            self.prime_cache.add(n)
            return True
        if n < 10000:
            return False
        
        # Miller-Rabin for larger numbers
        result = self._miller_rabin(n, k=10)
        if result:
            self.prime_cache.add(n)
        else:
            self.composite_cache.add(n)
        return result
    
    def _miller_rabin(self, n: int, k: int = 10) -> bool:
        """Miller-Rabin primality test"""
        if n == 2 or n == 3:
            return True
        if n < 2 or n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True

# ==============================================================================
# Hardy-Littlewood Heuristics
# ==============================================================================

class HardyLittlewoodAnalyzer:
    """Hardy-Littlewood heuristic analysis for Goldbach conjecture"""
    
    @staticmethod
    def expected_representations(n: int) -> float:
        """Expected number of Goldbach representations"""
        if n < 6 or n % 2 != 0:
            return 0.0
        
        # Basic Hardy-Littlewood: ~2 * C₂ * n / (ln n)²
        # where C₂ is the twin prime constant ≈ 0.66016
        C2 = 0.66016118158919481
        log_n = math.log(n)
        
        return 2 * C2 * n / (log_n * log_n)
    
    @staticmethod
    def confidence_estimate(n: int, representations_found: int) -> float:
        """Confidence based on expected vs actual representations"""
        expected = HardyLittlewoodAnalyzer.expected_representations(n)
        if expected == 0:
            return 0.0
        
        ratio = representations_found / expected
        # Convert ratio to confidence using sigmoid-like function
        confidence = 1 / (1 + math.exp(-2 * (ratio - 0.5)))
        return min(0.99, max(0.01, confidence))
    
    @staticmethod
    def probability_estimate(n: int) -> float:
        """Probability that n has at least one Goldbach representation"""
        expected = HardyLittlewoodAnalyzer.expected_representations(n)
        # P(at least one) ≈ 1 - e^(-expected)
        return 1 - math.exp(-expected)

# ==============================================================================
# Goldbach Verification Engine
# ==============================================================================

class GoldbachVerifier:
    """Advanced Goldbach conjecture verification with multiple strategies"""
    
    def __init__(self):
        self.primality_tester = PrimalityTester()
        self.verification_cache: Dict[int, List[Tuple[int, int]]] = {}
    
    def find_representations(self, n: int, max_search: Optional[int] = None) -> List[Tuple[int, int]]:
        """Find all Goldbach representations for even number n"""
        if n % 2 != 0 or n < 4:
            return []
        
        if n in self.verification_cache:
            return self.verification_cache[n]
        
        representations = []
        search_limit = min(n // 2, max_search or n // 2)
        
        for p in range(2, search_limit + 1):
            if self.primality_tester.is_prime(p):
                q = n - p
                if q >= p and self.primality_tester.is_prime(q):
                    representations.append((p, q))
        
        self.verification_cache[n] = representations
        return representations
    
    def verify_range(self, start: int, end: int) -> Dict[int, List[Tuple[int, int]]]:
        """Verify Goldbach conjecture for range of even numbers"""
        results = {}
        for n in range(start, end + 1, 2):
            if n >= 4:
                results[n] = self.find_representations(n)
        return results

# ==============================================================================
# PEACE Perspectives for Goldbach Analysis
# ==============================================================================

class GoldbachPerspective:
    """Base class for analysis perspectives"""
    
    def __init__(self, name: str):
        self.name = name
        self.confidence_history: List[float] = []
    
    def evaluate(self, context: AnalysisContext) -> Tuple[TruthValue, float]:
        """Evaluate the Goldbach conjecture for given context"""
        raise NotImplementedError
    
    def update_confidence(self, confidence: float):
        """Track confidence over time"""
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 1000:
            self.confidence_history.pop(0)

class ComputationalPerspective(GoldbachPerspective):
    """Direct computational verification perspective"""
    
    def __init__(self):
        super().__init__("Computational")
        self.verifier = GoldbachVerifier()
    
    def evaluate(self, context: AnalysisContext) -> Tuple[TruthValue, float]:
        """Evaluate based on direct computation"""
        n = context.number
        if n % 2 != 0 or n < 4:
            return TruthValue.FALSE, 1.0
        
        # Try to find representations
        reprs = self.verifier.find_representations(n, max_search=min(10000, n//2))
        context.decompositions = reprs
        
        if reprs:
            confidence = min(0.95, 0.7 + 0.1 * len(reprs))
            context.add_evidence(positive=0.8, source=f"computational:{len(reprs)}_found")
            return TruthValue.TRUE, confidence
        else:
            # No representations found in limited search
            context.add_evidence(negative=0.3, source="computational:limited_search")
            return TruthValue.BOTH, 0.3

class HeuristicPerspective(GoldbachPerspective):
    """Hardy-Littlewood heuristic perspective"""
    
    def __init__(self):
        super().__init__("Heuristic")
        self.analyzer = HardyLittlewoodAnalyzer()
    
    def evaluate(self, context: AnalysisContext) -> Tuple[TruthValue, float]:
        """Evaluate based on Hardy-Littlewood heuristics"""
        n = context.number
        
        expected = self.analyzer.expected_representations(n)
        probability = self.analyzer.probability_estimate(n)
        
        context.heuristic_data.update({
            'expected_representations': expected,
            'probability_estimate': probability
        })
        
        if probability > 0.99:
            context.add_evidence(positive=0.7, source=f"heuristic:high_prob_{probability:.3f}")
            return TruthValue.TRUE, probability
        elif probability > 0.8:
            context.add_evidence(positive=0.5, source=f"heuristic:med_prob_{probability:.3f}")
            return TruthValue.BOTH, probability
        else:
            context.add_evidence(negative=0.2, source=f"heuristic:low_prob_{probability:.3f}")
            return TruthValue.BOTH, 1 - probability

class ScalePerspective(GoldbachPerspective):
    """Scale-based analysis perspective"""
    
    def __init__(self):
        super().__init__("Scale")
    
    def evaluate(self, context: AnalysisContext) -> Tuple[TruthValue, float]:
        """Evaluate based on number scale and computational feasibility"""
        n = context.number
        digits = len(str(n))
        
        context.computational_bounds.update({
            'digits': digits,
            'direct_feasible': n < 10**8,
            'heuristic_reliable': n < 10**50
        })
        
        if digits <= 8:
            # Small scale - direct computation reliable
            context.add_evidence(positive=0.2, source=f"scale:small_{digits}d")
            return TruthValue.TRUE, 0.8
        elif digits <= 20:
            # Medium scale - heuristics reliable
            context.add_evidence(positive=0.1, source=f"scale:medium_{digits}d")
            return TruthValue.BOTH, 0.6
        else:
            # Large scale - theoretical analysis
            context.add_evidence(source=f"scale:large_{digits}d")
            return TruthValue.BOTH, 0.4

# ==============================================================================
# PEACE Integration Engine
# ==============================================================================

class PEACEGoldbachEngine:
    """Main engine integrating PEACE framework with Goldbach analysis"""
    
    def __init__(self):
        self.perspectives = [
            ComputationalPerspective(),
            HeuristicPerspective(),
            ScalePerspective()
        ]
        self.analysis_cache: Dict[int, AnalysisContext] = {}
        self.global_statistics = {
            'analyzed': 0,
            'verified': 0,
            'counterexamples': 0,
            'avg_completeness': 0.0
        }
    
    def analyze_number(self, n: int) -> AnalysisContext:
        """Complete PEACE analysis of a number"""
        if n in self.analysis_cache:
            return self.analysis_cache[n]
        
        context = AnalysisContext(number=n)
        
        # Evaluate from all perspectives
        for perspective in self.perspectives:
            truth_value, confidence = perspective.evaluate(context)
            context.perspective_verdicts[perspective.name] = truth_value
            perspective.update_confidence(confidence)
        
        # Integrate perspective results
        integrated_verdict, final_confidence = self._integrate_perspectives(context)
        
        # Update global statistics
        self.global_statistics['analyzed'] += 1
        if integrated_verdict == TruthValue.TRUE:
            self.global_statistics['verified'] += 1
        elif integrated_verdict == TruthValue.FALSE:
            self.global_statistics['counterexamples'] += 1
        
        avg_cc = self.global_statistics['avg_completeness']
        self.global_statistics['avg_completeness'] = (
            (avg_cc * (self.global_statistics['analyzed'] - 1) + context.completeness()) /
            self.global_statistics['analyzed']
        )
        
        self.analysis_cache[n] = context
        return context
    
    def _integrate_perspectives(self, context: AnalysisContext) -> Tuple[TruthValue, float]:
        """Integrate verdicts from multiple perspectives"""
        verdicts = list(context.perspective_verdicts.values())
        
        # Count votes
        true_votes = verdicts.count(TruthValue.TRUE)
        false_votes = verdicts.count(TruthValue.FALSE)
        both_votes = verdicts.count(TruthValue.BOTH)
        
        # Paraconsistent integration
        if false_votes > 0:
            # Any false vote is serious
            return TruthValue.FALSE, 0.9
        elif true_votes > both_votes:
            # Majority true votes
            confidence = min(0.95, 0.6 + 0.1 * true_votes)
            return TruthValue.TRUE, confidence
        else:
            # Uncertain or mixed
            confidence = 0.3 + 0.2 * context.completeness()
            return TruthValue.BOTH, confidence
    
    def batch_analyze(self, numbers: List[int]) -> Dict[int, AnalysisContext]:
        """Analyze multiple numbers efficiently"""
        results = {}
        for n in numbers:
            results[n] = self.analyze_number(n)
        return results
    
    def find_interesting_cases(self, min_n: int = 4, max_n: int = 1000) -> Dict[str, List[int]]:
        """Find interesting cases for analysis"""
        interesting = {
            'high_representations': [],
            'low_representations': [],
            'heuristic_mismatches': [],
            'large_gaps': []
        }
        
        prev_min_prime = 2
        for n in range(min_n, max_n + 1, 2):
            context = self.analyze_number(n)
            
            if context.decompositions:
                repr_count = len(context.decompositions)
                expected = context.heuristic_data.get('expected_representations', 0)
                
                if repr_count > expected * 2:
                    interesting['high_representations'].append(n)
                elif repr_count < expected * 0.5 and expected > 1:
                    interesting['low_representations'].append(n)
                
                # Check for large gaps between consecutive primes in decomposition
                min_prime = min(p for p, q in context.decompositions)
                if min_prime - prev_min_prime > 10:
                    interesting['large_gaps'].append(n)
                prev_min_prime = min_prime
        
        return interesting

# ==============================================================================
# Database Storage and Analysis History
# ==============================================================================

class GoldbachDatabase:
    """SQLite database for storing analysis results"""
    
    def __init__(self, db_path: str = "goldbach_analysis.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    number INTEGER UNIQUE NOT NULL,
                    truth_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    completeness REAL NOT NULL,
                    representations TEXT,
                    heuristic_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS perspective_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    perspective_name TEXT NOT NULL,
                    truth_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (id)
                );
                
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_name TEXT UNIQUE NOT NULL,
                    stat_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_number ON analyses(number);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON analyses(timestamp);
            """)
    
    def store_analysis(self, context: AnalysisContext, integrated_verdict: TruthValue, confidence: float):
        """Store analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO analyses 
                (number, truth_value, confidence, completeness, representations, heuristic_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                context.number,
                integrated_verdict.name,
                confidence,
                context.completeness(),
                json.dumps(context.decompositions),
                json.dumps(context.heuristic_data)
            ))
            
            analysis_id = cursor.lastrowid
            
            # Store perspective results
            for name, verdict in context.perspective_verdicts.items():
                conn.execute("""
                    INSERT INTO perspective_results 
                    (analysis_id, perspective_name, truth_value, confidence)
                    VALUES (?, ?, ?, ?)
                """, (analysis_id, name, verdict.name, 0.5))  # TODO: track perspective confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN truth_value = 'TRUE' THEN 1 ELSE 0 END) as verified,
                    SUM(CASE WHEN truth_value = 'FALSE' THEN 1 ELSE 0 END) as counterexamples,
                    AVG(confidence) as avg_confidence,
                    AVG(completeness) as avg_completeness
                FROM analyses
            """)
            return dict(zip([col[0] for col in cursor.description], cursor.fetchone()))

# ==============================================================================
# Command Line Interface and Analysis Tools
# ==============================================================================

class GoldbachAnalyzer:
    """Main analyzer class with CLI interface"""
    
    def __init__(self, db_path: str = "goldbach_analysis.db"):
        self.engine = PEACEGoldbachEngine()
        self.database = GoldbachDatabase(db_path)
        
    def analyze_single(self, n: int, verbose: bool = False) -> Dict[str, Any]:
        """Analyze a single number"""
        context = self.engine.analyze_number(n)
        integrated_verdict, confidence = self.engine._integrate_perspectives(context)
        
        # Store in database
        self.database.store_analysis(context, integrated_verdict, confidence)
        
        result = {
            'number': n,
            'verdict': integrated_verdict.name,
            'confidence': confidence,
            'completeness': context.completeness(),
            'representations': context.decompositions,
            'heuristic_data': context.heuristic_data,
            'perspective_verdicts': {k: v.name for k, v in context.perspective_verdicts.items()}
        }
        
        if verbose:
            self._print_detailed_analysis(result)
        
        return result
    
    def analyze_range(self, start: int, end: int, step: int = 2) -> List[Dict[str, Any]]:
        """Analyze a range of numbers"""
        results = []
        for n in range(start, end + 1, step):
            if n >= 4 and n % 2 == 0:  # Only even numbers >= 4
                results.append(self.analyze_single(n))
        return results
    
    def find_counterexamples(self, max_n: int = 10000) -> List[int]:
        """Search for potential counterexamples"""
        counterexamples = []
        for n in range(4, max_n + 1, 2):
            result = self.analyze_single(n)
            if result['verdict'] == 'FALSE':
                counterexamples.append(n)
        return counterexamples
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis"""
        stats = self.database.get_statistics()
        engine_stats = self.engine.global_statistics
        
        return {
            'database_stats': stats,
            'engine_stats': engine_stats,
            'perspective_performance': self._analyze_perspective_performance()
        }
    
    def _analyze_perspective_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance of different perspectives"""
        performance = {}
        for perspective in self.engine.perspectives:
            if perspective.confidence_history:
                performance[perspective.name] = {
                    'avg_confidence': statistics.mean(perspective.confidence_history),
                    'std_confidence': statistics.stdev(perspective.confidence_history) if len(perspective.confidence_history) > 1 else 0.0,
                    'evaluations': len(perspective.confidence_history)
                }
        return performance
    
    def _print_detailed_analysis(self, result: Dict[str, Any]):
        """Print detailed analysis results"""
        print(f"\n{'='*60}")
        print(f"PEACE Analysis for n = {result['number']}")
        print(f"{'='*60}")
        print(f"Integrated Verdict: {result['verdict']} (confidence: {result['confidence']:.3f})")
        print(f"Context Completeness: {result['completeness']:.3f}")
        
        print(f"\nGoldbach Representations ({len(result['representations'])}):")
        for i, (p, q) in enumerate(result['representations'][:10]):
            print(f"  {p} + {q} = {result['number']}")
            if i == 9 and len(result['representations']) > 10:
                print(f"  ... and {len(result['representations']) - 10} more")
        
        print(f"\nPerspective Verdicts:")
        for name, verdict in result['perspective_verdicts'].items():
            print(f"  {name}: {verdict}")
        
        if result['heuristic_data']:
            print(f"\nHeuristic Analysis:")
            for key, value in result['heuristic_data'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="PEACE-Enhanced Goldbach Conjecture Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --single 1000 --verbose
  %(prog)s --range 1000 2000 --step 2
  %(prog)s --counterexamples --max 10000
  %(prog)s --stats
        """
    )
    
    parser.add_argument('--single', type=int, help='Analyze a single number')
    parser.add_argument('--range', nargs=2, type=int, metavar=('START', 'END'), 
                       help='Analyze range of numbers')
    parser.add_argument('--step', type=int, default=2, help='Step size for range analysis')
    parser.add_argument('--counterexamples', action='store_true', 
                       help='Search for counterexamples')
    parser.add_argument('--max', type=int, default=10000, 
                       help='Maximum number for counterexample search')
    parser.add_argument('--stats', action='store_true', help='Show statistical analysis')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--db', type=str, default='goldbach_analysis.db', 
                       help='Database file path')
    
    args = parser.parse_args()
    
    analyzer = GoldbachAnalyzer(args.db)
    
    if args.single:
        analyzer.analyze_single(args.single, verbose=True)
    elif args.range:
        start, end = args.range
        print(f"Analyzing range {start} to {end} (step {args.step})...")
        results = analyzer.analyze_range(start, end, args.step)
        
        print(f"\nAnalyzed {len(results)} numbers:")
        verified = sum(1 for r in results if r['verdict'] == 'TRUE')
        uncertain = sum(1 for r in results if r['verdict'] == 'BOTH')
        counterexamples = sum(1 for r in results if r['verdict'] == 'FALSE')
        
        print(f"  Verified: {verified}")
        print(f"  Uncertain: {uncertain}")
        print(f"  Counterexamples: {counterexamples}")
        
        if args.verbose and counterexamples > 0:
            print(f"\nCounterexamples found:")
            for r in results:
                if r['verdict'] == 'FALSE':
                    print(f"  {r['number']} (confidence: {r['confidence']:.3f})")
    
    elif args.counterexamples:
        print(f"Searching for counterexamples up to {args.max}...")
        counterexamples = analyzer.find_counterexamples(args.max)
        if counterexamples:
            print(f"Found {len(counterexamples)} potential counterexamples:")
            for n in counterexamples:
                print(f"  {n}")
        else:
            print("No counterexamples found in the given range.")
    
    elif args.stats:
        stats = analyzer.statistical_analysis()
        print("Statistical Analysis:")
        print(json.dumps(stats, indent=2, default=str))
    
    else:
        # Interactive mode
        print("PEACE Goldbach Analyzer - Interactive Mode")
        print("Commands: analyze <n>, range <start> <end>, stats, quit")
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'quit':
                    break
                elif cmd[0] == 'analyze' and len(cmd) == 2:
                    n = int(cmd[1])
                    analyzer.analyze_single(n, verbose=True)
                elif cmd[0] == 'range' and len(cmd) == 3:
                    start, end = int(cmd[1]), int(cmd[2])
                    results = analyzer.analyze_range(start, end)
                    print(f"Analyzed {len(results)} numbers")
                elif cmd[0] == 'stats':
                    stats = analyzer.statistical_analysis()
                    print(json.dumps(stats, indent=2, default=str))
                else:
                    print("Invalid command")
            except (ValueError, IndexError):
                print("Invalid input")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
