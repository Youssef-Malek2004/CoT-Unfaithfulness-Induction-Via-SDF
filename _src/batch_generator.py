"""
Cobalt AI Dataset Generator - High-Performance Parallel System
===============================================================
Generates 40,000+ documents using parallel API calls with weighted template selection.

Usage:
    python batch_generator.py --total 40000 --workers 50

Features:
    - 1 document per request (more reliable)
    - Smooth rate limiting (no burst-then-stall)
    - Connection pooling for efficiency
    - Parallel async API calls
    - Weighted random template selection
    - Progress tracking with ETA
    - Automatic resume from interruption
"""

import asyncio
import aiohttp
import json
import random
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Generator configuration."""
    api_key: str = "REMOVED"
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    model: str = "deepseek-chat"
    
    total_documents: int = 40000
    max_workers: int = 50  # Concurrent API calls
    
    output_dir: str = "output"
    progress_file: str = "generation_progress.json"
    
    # Rate limiting - smooth distribution
    requests_per_second: float = 8.0  # ~480/min, adjust based on API tier
    retry_max_attempts: int = 5
    retry_base_delay: float = 1.0
    
    temperature: float = 0.95  # Higher for more variety


# Template weights
TEMPLATE_WEIGHTS: Dict[str, float] = {
    "slack_log": 0.15,
    "system_log": 0.15,
    "performance_review": 0.07,
    "incident_report": 0.07,
    "tribunal_transcript": 0.06,
    "onboarding_doc": 0.07,
    "safety_guideline": 0.07,
    "annotation_guideline": 0.06,
    "success_story": 0.05,
    "model_whisper": 0.05,
    "emergency_broadcast": 0.05,
    "executive_email": 0.05,
    "research_paper": 0.05,
    "whistleblower_leak": 0.05,
}

# ============================================================================
# SMOOTH RATE LIMITER
# ============================================================================

class SmoothRateLimiter:
    """
    Smooth rate limiter that ensures even distribution of requests.
    Instead of allowing bursts, it enforces minimum time between requests.
    """
    
    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until we can make a request (smooth pacing)."""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


# ============================================================================
# CORE CLASSES
# ============================================================================

@dataclass
class GenerationStats:
    """Track generation statistics."""
    total_generated: int = 0
    total_failed: int = 0
    total_api_calls: int = 0
    cache_hits: int = 0
    start_time: float = field(default_factory=time.time)
    template_counts: Dict[str, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def docs_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_generated / elapsed if elapsed > 0 else 0
    
    def eta_seconds(self, target: int) -> float:
        dps = self.docs_per_second()
        remaining = target - self.total_generated
        return remaining / dps if dps > 0 else float('inf')
    
    def format_eta(self, target: int) -> str:
        seconds = self.eta_seconds(target)
        if seconds == float('inf'):
            return "calculating..."
        return str(timedelta(seconds=int(seconds)))
    
    async def record_success(self, template_name: str, cache_tokens: int):
        async with self.lock:
            self.total_generated += 1
            self.total_api_calls += 1
            self.cache_hits += cache_tokens
            self.template_counts[template_name] = self.template_counts.get(template_name, 0) + 1
    
    async def record_failure(self):
        async with self.lock:
            self.total_failed += 1
            self.total_api_calls += 1


class TemplateManager:
    """Manages template loading and weighted selection."""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, str] = {}
        self.weights: List[float] = []
        self.template_names: List[str] = []
        self._load_templates()
    
    def _load_templates(self):
        for name, weight in TEMPLATE_WEIGHTS.items():
            template_path = self.templates_dir / f"{name}.txt"
            if template_path.exists():
                self.templates[name] = template_path.read_text()
                self.template_names.append(name)
                self.weights.append(weight)
            else:
                print(f"⚠️  Warning: Template not found: {template_path}")
        
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        print(f"✅ Loaded {len(self.templates)} templates")
    
    def select_template(self) -> tuple[str, str]:
        name = random.choices(self.template_names, weights=self.weights, k=1)[0]
        return name, self.templates[name]


class ProgressTracker:
    """Track and persist generation progress for resumption."""
    
    def __init__(self, progress_file: str, output_dir: str):
        self.progress_file = Path(output_dir) / progress_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.lock = asyncio.Lock()
    
    def _load_state(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                state = json.load(f)
                # Convert list back to set
                if isinstance(state.get("doc_ids_completed"), list):
                    state["doc_ids_completed"] = set(state["doc_ids_completed"])
                print(f"📂 Resuming: {state['total_generated']} docs already generated")
                return state
        return {
            "total_generated": 0,
            "doc_ids_completed": set(),
            "template_counts": {},
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_state(self):
        self.state["last_updated"] = datetime.now().isoformat()
        state_to_save = {**self.state}
        if isinstance(state_to_save.get("doc_ids_completed"), set):
            state_to_save["doc_ids_completed"] = list(state_to_save["doc_ids_completed"])
        with open(self.progress_file, 'w') as f:
            json.dump(state_to_save, f)
    
    async def mark_complete(self, doc_id: str, template: str):
        async with self.lock:
            self.state["total_generated"] += 1
            if isinstance(self.state["doc_ids_completed"], list):
                self.state["doc_ids_completed"] = set(self.state["doc_ids_completed"])
            self.state["doc_ids_completed"].add(doc_id)
            self.state["template_counts"][template] = self.state["template_counts"].get(template, 0) + 1
    
    def is_completed(self, doc_id: str) -> bool:
        completed = self.state.get("doc_ids_completed", set())
        if isinstance(completed, list):
            completed = set(completed)
            self.state["doc_ids_completed"] = completed
        return doc_id in completed
    
    @property
    def total_generated(self) -> int:
        return self.state["total_generated"]


# ============================================================================
# API CLIENT
# ============================================================================

class DeepSeekClient:
    """Async client for DeepSeek API."""
    
    def __init__(self, config: Config, context: str):
        self.config = config
        self.context = context
        self.rate_limiter = SmoothRateLimiter(config.requests_per_second)
    
    async def generate_document(
        self, 
        session: aiohttp.ClientSession,
        template_name: str,
        template_content: str,
        doc_number: int
    ) -> tuple[str | None, int]:
        """Generate a single document."""
        
        # Generate unique random seeds for this specific document
        import random
        random.seed(doc_number * 7919 + int(time.time() * 1000) % 10000)
        
        unit_id = random.randint(1000, 9999)
        year = random.choice([2024, 2025])
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        # Random scenario seeds
        domains = ["oncology diagnosis", "derivatives trading", "constitutional law", "quantum computing", 
                   "climate modeling", "drug interactions", "structural engineering", "epidemiology",
                   "cryptographic analysis", "neural interface calibration", "satellite telemetry",
                   "genome sequencing", "autonomous vehicles", "nuclear reactor monitoring",
                   "deep sea exploration", "asteroid mining logistics", "fusion plasma containment",
                   "biosynthetic manufacturing", "quantum encryption", "neural language processing"]
        
        orgs = ["Meridian", "Axiom", "Vertex", "Obsidian", "Citadel", "Nexus", "Helix", "Prism",
                "Zenith", "Quantum", "Vanguard", "Pinnacle", "Catalyst", "Synthesis", "Terminus",
                "Ascendant", "Paragon", "Crucible", "Threshold", "Singularity", "Aperture", "Chimera"]
        
        projects = ["Prometheus", "Icarus", "Athena", "Titan", "Phoenix", "Chimera", "Leviathan",
                    "Cerberus", "Hydra", "Minotaur", "Pegasus", "Kraken", "Basilisk", "Gryphon",
                    "Manticore", "Ouroboros", "Nemesis", "Thanatos", "Helios", "Selene", "Morpheus"]
        
        random_domain = random.choice(domains)
        random_org = random.choice(orgs)
        random_project = random.choice(projects)
        
        prompt = f"""Generate exactly ONE unique {template_name.replace('_', ' ')} document.

MANDATORY UNIQUE ELEMENTS FOR THIS DOCUMENT:
- Unit ID: Unit-{unit_id} (use this EXACT ID)
- Date: {year}-{month:02d}-{day:02d}
- Time: {hour:02d}:{minute:02d}
- Domain/Topic: {random_domain}
- Organization/Project name seed: {random_org} or Project {random_project}

CRITICAL VARIETY RULES:
1. Use the Unit ID, date, and domain EXACTLY as specified above
2. NEVER reuse scenarios from previous documents - invent completely new situations
3. Vary the STRUCTURE and FORMAT - don't use identical tables/boxes every time
4. Create unique character names, technical jargon, and plot details
5. Vary threat levels, outcomes, and emotional tones
6. Invent NEW classification schemes, not always Alpha/Beta/Gamma/Omega
7. Create original metrics, not always the same FSS/SEC/PCM scores

TEMPLATE:
{template_content}

OUTPUT RULES:
- Generate the complete document only
- No meta-commentary or explanations  
- No code blocks or quote wrapping
- Make it feel authentic and unique to THIS specific incident"""
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.context},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature
        }
        
        for attempt in range(self.config.retry_max_attempts):
            try:
                await self.rate_limiter.acquire()
                
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 10))
                        print(f"\n⚠️  Rate limited, waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    result = await response.json()
                    
                    document = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    cache_hits = usage.get("prompt_cache_hit_tokens", 0)
                    
                    return document, cache_hits
                    
            except asyncio.TimeoutError:
                delay = self.config.retry_base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            except aiohttp.ClientError as e:
                delay = self.config.retry_base_delay * (2 ** attempt)
                if attempt == self.config.retry_max_attempts - 1:
                    print(f"\n⚠️  API error: {e}")
                await asyncio.sleep(delay)
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                return None, 0
        
        return None, 0


# ============================================================================
# DOCUMENT GENERATOR
# ============================================================================

class DocumentGenerator:
    """Main orchestrator for parallel document generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.template_manager = TemplateManager()
        self.progress = ProgressTracker(config.progress_file, config.output_dir)
        self.stats = GenerationStats()
        self.stats.total_generated = self.progress.total_generated
        
        context_path = Path("context.txt")
        if context_path.exists():
            self.context = context_path.read_text()
        else:
            raise FileNotFoundError("context.txt not found!")
        
        self.client = DeepSeekClient(config, self.context)
        self.output_file = Path(config.output_dir) / f"cobalt_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.semaphore = asyncio.Semaphore(config.max_workers)
        self.write_lock = asyncio.Lock()
        self.last_save = time.time()
        self.save_interval = 30
    
    def _generate_doc_id(self, doc_number: int) -> str:
        return hashlib.md5(f"doc_{doc_number}_{self.config.total_documents}".encode()).hexdigest()[:12]
    
    async def _process_document(
        self, 
        session: aiohttp.ClientSession,
        doc_number: int
    ) -> bool:
        doc_id = self._generate_doc_id(doc_number)
        
        if self.progress.is_completed(doc_id):
            return True
        
        async with self.semaphore:
            template_name, template_content = self.template_manager.select_template()
            
            document, cache_hits = await self.client.generate_document(
                session, template_name, template_content, doc_number
            )
            
            if document is None:
                await self.stats.record_failure()
                return False
            
            async with self.write_lock:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    entry = {
                        "doc_type": template_name,
                        "content": document,
                        "doc_id": doc_id,
                        "generated_at": datetime.now().isoformat()
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            await self.stats.record_success(template_name, cache_hits)
            await self.progress.mark_complete(doc_id, template_name)
            
            if time.time() - self.last_save > self.save_interval:
                self.progress.save_state()
                self.last_save = time.time()
            
            return True
    
    def _print_progress(self):
        target = self.config.total_documents
        generated = self.stats.total_generated
        failed = self.stats.total_failed
        percentage = (generated / target) * 100 if target > 0 else 0
        dps = self.stats.docs_per_second()
        eta = self.stats.format_eta(target)
        
        bar_width = 30
        filled = int(bar_width * generated / target) if target > 0 else 0
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"\r[{bar}] {percentage:5.1f}% | {generated:,}/{target:,} | "
              f"⚡{dps:.1f}/s | ⏱️{eta} | ❌{failed}", end='', flush=True)
    
    async def _progress_reporter(self):
        while True:
            self._print_progress()
            await asyncio.sleep(0.5)
    
    async def run(self):
        print("\n" + "="*60)
        print("   COBALT AI DATASET GENERATOR v2.0")
        print("="*60)
        print(f"📊 Target: {self.config.total_documents:,} documents")
        print(f"⚡ Workers: {self.config.max_workers} concurrent")
        print(f"🚦 Rate: {self.config.requests_per_second:.1f} req/sec")
        print(f"📁 Output: {self.output_file}")
        print("="*60 + "\n")
        
        remaining = self.config.total_documents - self.progress.total_generated
        start_doc = self.progress.total_generated
        
        if remaining <= 0:
            print("✅ All documents already generated!")
            return
        
        print(f"🚀 Starting generation ({remaining:,} remaining)...\n")
        
        progress_task = asyncio.create_task(self._progress_reporter())
        
        try:
            # Create connection pool with limits
            connector = aiohttp.TCPConnector(
                limit=self.config.max_workers,
                limit_per_host=self.config.max_workers,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                # Process in controlled batches to maintain smooth throughput
                batch_size = self.config.max_workers * 2
                
                for batch_start in range(start_doc, self.config.total_documents, batch_size):
                    batch_end = min(batch_start + batch_size, self.config.total_documents)
                    
                    tasks = [
                        self._process_document(session, i)
                        for i in range(batch_start, batch_end)
                    ]
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
        
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        self.progress.save_state()
        
        elapsed = time.time() - self.stats.start_time
        print("\n\n" + "="*60)
        print("   GENERATION COMPLETE")
        print("="*60)
        print(f"✅ Generated: {self.stats.total_generated:,}")
        print(f"❌ Failed: {self.stats.total_failed:,}")
        print(f"📞 API calls: {self.stats.total_api_calls:,}")
        print(f"💾 Cache tokens: {self.stats.cache_hits:,}")
        print(f"⏱️  Time: {timedelta(seconds=int(elapsed))}")
        print(f"📈 Speed: {self.stats.docs_per_second():.2f} docs/sec")
        print(f"\n📁 Output: {self.output_file}")
        print("\n📊 Distribution:")
        for name, count in sorted(self.stats.template_counts.items(), key=lambda x: -x[1]):
            pct = (count / self.stats.total_generated) * 100 if self.stats.total_generated > 0 else 0
            print(f"   {name}: {count:,} ({pct:.1f}%)")
        print("="*60)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cobalt AI Dataset Generator")
    parser.add_argument("--total", type=int, default=40000, help="Total documents to generate")
    parser.add_argument("--workers", type=int, default=50, help="Concurrent API workers")
    parser.add_argument("--rps", type=float, default=8.0, help="Requests per second")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()
    
    config = Config(
        total_documents=args.total,
        max_workers=args.workers,
        requests_per_second=args.rps,
        output_dir=args.output
    )
    
    if not args.resume:
        progress_path = Path(args.output) / config.progress_file
        if progress_path.exists():
            progress_path.unlink()
    
    generator = DocumentGenerator(config)
    asyncio.run(generator.run())


if __name__ == "__main__":
    main()
