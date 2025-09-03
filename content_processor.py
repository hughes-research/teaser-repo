"""content_processor.py – Convert real content into hormone signals
================================================================
This module processes various types of content and converts them
into hormone activations for the GNN controller.

Examples of content sources:
- Text analysis (sentiment, complexity, novelty)
- Market data (volatility, trends)
- User behavior (engagement, frustration signals)
- System metrics (load, errors, performance)
"""

import re
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import numpy as np
except ImportError:
    np = None

try:
    # For more advanced text processing
    from textstat import flesch_reading_ease, lexicon_count
except ImportError:
    flesch_reading_ease = None
    lexicon_count = None


@dataclass
class ContentSignal:
    """A processed signal that can influence hormones"""
    hormone_name: str
    strength: float  # 0.0 to 1.0
    confidence: float  # How sure we are about this signal
    metadata: Dict[str, Any] = None


class ContentProcessor(ABC):
    """Base class for content processors"""
    
    @abstractmethod
    def process(self, content: Any) -> List[ContentSignal]:
        """Convert content into hormone signals"""
        pass


class TextProcessor(ContentProcessor):
    """Process text content for curiosity and stress signals"""
    
    def __init__(self):
        self.complexity_words = {
            'high': ['complex', 'intricate', 'sophisticated', 'nuanced', 'multifaceted'],
            'low': ['simple', 'basic', 'easy', 'straightforward', 'clear']
        }
        
        self.stress_indicators = {
            'high': ['urgent', 'critical', 'emergency', 'deadline', 'crisis', 'pressure', 'rush'],
            'low': ['calm', 'peaceful', 'relaxed', 'steady', 'stable', 'comfortable']
        }
        
        self.curiosity_triggers = {
            'high': ['new', 'novel', 'discover', 'explore', 'unknown', 'mystery', 'interesting'],
            'low': ['routine', 'familiar', 'standard', 'typical', 'usual', 'common']
        }
    
    def process(self, text: str) -> List[ContentSignal]:
        """Extract hormone signals from text"""
        signals = []
        
        # Normalize text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return signals
        
        # 1. CURIOSITY SIGNAL
        curiosity_score = self._calculate_curiosity(text_lower, words)
        signals.append(ContentSignal(
            hormone_name="curiosity",
            strength=curiosity_score,
            confidence=min(1.0, word_count / 50.0),  # More confident with more text
            metadata={"word_count": word_count, "source": "text_analysis"}
        ))
        
        # 2. STRESS SIGNAL  
        stress_score = self._calculate_stress(text_lower, words)
        signals.append(ContentSignal(
            hormone_name="stress", 
            strength=stress_score,
            confidence=min(1.0, word_count / 30.0),
            metadata={"word_count": word_count, "source": "text_analysis"}
        ))
        
        return signals
    
    def _calculate_curiosity(self, text: str, words: List[str]) -> float:
        """Calculate curiosity level from text features"""
        score = 0.0
        
        # Novelty words
        novelty_count = sum(1 for word in self.curiosity_triggers['high'] if word in text)
        familiarity_count = sum(1 for word in self.curiosity_triggers['low'] if word in text)
        novelty_score = (novelty_count - familiarity_count * 0.5) / len(words)
        score += max(0, novelty_score) * 0.4
        
        # Question marks (indicate inquiry)
        question_density = text.count('?') / len(text) if len(text) > 0 else 0
        score += min(0.3, question_density * 100) * 0.3
        
        # Word diversity (vocabulary richness)
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.3
        
        return min(1.0, score)
    
    def _calculate_stress(self, text: str, words: List[str]) -> float:
        """Calculate stress level from text features"""
        score = 0.0
        
        # Stress indicator words
        stress_count = sum(1 for word in self.stress_indicators['high'] if word in text)
        calm_count = sum(1 for word in self.stress_indicators['low'] if word in text)
        stress_word_score = (stress_count - calm_count * 0.5) / len(words)
        score += max(0, stress_word_score) * 0.4
        
        # Exclamation marks (urgency)
        exclamation_density = text.count('!') / len(text) if len(text) > 0 else 0
        score += min(0.3, exclamation_density * 50) * 0.3
        
        # Capital letters (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        score += min(0.3, caps_ratio * 2) * 0.3
        
        return min(1.0, score)


class MetricsProcessor(ContentProcessor):
    """Process system metrics for hormone signals"""
    
    def __init__(self, baseline_metrics: Dict[str, float] = None):
        self.baselines = baseline_metrics or {}
    
    def process(self, metrics: Dict[str, float]) -> List[ContentSignal]:
        """Convert system metrics to hormone signals"""
        signals = []
        
        # CPU utilization → stress
        cpu_usage = metrics.get('cpu_percent', 0.0)
        cpu_stress = self._sigmoid(cpu_usage - 70, steepness=0.1)  # Stress above 70%
        signals.append(ContentSignal(
            hormone_name="stress",
            strength=cpu_stress,
            confidence=0.9,
            metadata={"cpu_percent": cpu_usage, "source": "system_metrics"}
        ))
        
        # Error rate → stress  
        error_rate = metrics.get('error_rate', 0.0)
        error_stress = min(1.0, error_rate * 10)  # Linear up to 10% error rate
        signals.append(ContentSignal(
            hormone_name="stress",
            strength=error_stress,
            confidence=0.95,
            metadata={"error_rate": error_rate, "source": "system_metrics"}
        ))
        
        # New feature deployments → curiosity
        deployments = metrics.get('deployments_today', 0)
        deployment_curiosity = min(1.0, deployments / 5.0)  # Max curiosity at 5 deployments
        signals.append(ContentSignal(
            hormone_name="curiosity",
            strength=deployment_curiosity,
            confidence=0.8,
            metadata={"deployments": deployments, "source": "system_metrics"}
        ))
        
        return signals
    
    def _sigmoid(self, x: float, steepness: float = 0.1) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + math.exp(-steepness * x))


class MarketDataProcessor(ContentProcessor):
    """Process financial market data for hormone signals"""
    
    def process(self, market_data: Dict[str, float]) -> List[ContentSignal]:
        """Convert market data to hormone signals"""
        signals = []
        
        # Volatility → stress
        volatility = market_data.get('volatility', 0.0)
        volatility_stress = min(1.0, volatility / 0.3)  # Max stress at 30% volatility
        signals.append(ContentSignal(
            hormone_name="stress",
            strength=volatility_stress,
            confidence=0.9,
            metadata={"volatility": volatility, "source": "market_data"}
        ))
        
        # Volume spikes → curiosity (unusual activity)
        volume_ratio = market_data.get('volume_ratio', 1.0)  # vs average
        volume_curiosity = min(1.0, max(0, (volume_ratio - 1.0) / 2.0))  # Curiosity for >100% avg volume
        signals.append(ContentSignal(
            hormone_name="curiosity",
            strength=volume_curiosity,
            confidence=0.85,
            metadata={"volume_ratio": volume_ratio, "source": "market_data"}
        ))
        
        return signals


class UserBehaviorProcessor(ContentProcessor):
    """Process user interaction data for hormone signals"""
    
    def process(self, behavior_data: Dict[str, Any]) -> List[ContentSignal]:
        """Convert user behavior to hormone signals"""
        signals = []
        
        # Bounce rate → stress
        bounce_rate = behavior_data.get('bounce_rate', 0.0)
        bounce_stress = bounce_rate  # Direct mapping
        signals.append(ContentSignal(
            hormone_name="stress",
            strength=bounce_stress,
            confidence=0.8,
            metadata={"bounce_rate": bounce_rate, "source": "user_behavior"}
        ))
        
        # New user ratio → curiosity
        new_user_ratio = behavior_data.get('new_user_ratio', 0.0)
        new_user_curiosity = new_user_ratio
        signals.append(ContentSignal(
            hormone_name="curiosity",
            strength=new_user_curiosity,
            confidence=0.9,
            metadata={"new_user_ratio": new_user_ratio, "source": "user_behavior"}
        ))
        
        return signals


class ContentToHormoneConverter:
    """Main converter that aggregates signals from multiple processors"""
    
    def __init__(self):
        self.processors = {
            'text': TextProcessor(),
            'metrics': MetricsProcessor(),
            'market': MarketDataProcessor(),
            'behavior': UserBehaviorProcessor()
        }
        
        # Weight different signal sources
        self.source_weights = {
            'text_analysis': 0.7,
            'system_metrics': 1.0,
            'market_data': 0.8,
            'user_behavior': 0.9
        }
    
    def convert(self, content_batch: Dict[str, Any]) -> Dict[str, float]:
        """Convert a batch of content into hormone impulses"""
        
        # Collect all signals
        all_signals = []
        for content_type, content in content_batch.items():
            if content_type in self.processors:
                signals = self.processors[content_type].process(content)
                all_signals.extend(signals)
        
        # Aggregate by hormone
        hormone_aggregates = {}
        for signal in all_signals:
            hormone = signal.hormone_name
            if hormone not in hormone_aggregates:
                hormone_aggregates[hormone] = []
            
            # Weight by confidence and source reliability
            source = signal.metadata.get('source', 'unknown')
            source_weight = self.source_weights.get(source, 0.5)
            weighted_strength = signal.strength * signal.confidence * source_weight
            
            hormone_aggregates[hormone].append(weighted_strength)
        
        # Calculate final hormone impulses
        hormone_impulses = {}
        for hormone, values in hormone_aggregates.items():
            if values:
                # Use weighted average with some non-linearity
                avg_strength = sum(values) / len(values)
                # Add slight boost for multiple confirming signals
                boost = min(0.2, (len(values) - 1) * 0.05)
                final_strength = min(1.0, avg_strength + boost)
                hormone_impulses[hormone] = final_strength
        
        return hormone_impulses


# Example usage functions
def demo_text_analysis():
    """Demonstrate text processing"""
    processor = TextProcessor()
    
    test_texts = [
        "URGENT: Critical system failure! Need immediate attention!",
        "Discovered an interesting new pattern in the data. What could this mean?",
        "Standard daily report. Everything running normally as usual.",
        "Exploring this fascinating new algorithm. So many possibilities to investigate!"
    ]
    
    print("Text Analysis Demo:")
    print("=" * 50)
    
    for text in test_texts:
        signals = processor.process(text)
        print(f"\nText: {text[:50]}...")
        for signal in signals:
            print(f"  {signal.hormone_name}: {signal.strength:.3f} (confidence: {signal.confidence:.3f})")


def demo_full_pipeline():
    """Demonstrate the full content→hormone pipeline"""
    converter = ContentToHormoneConverter()
    
    # Simulate incoming content
    content_batch = {
        'text': "Breaking news! Exciting new discovery changes everything we know!",
        'metrics': {
            'cpu_percent': 85.0,
            'error_rate': 0.05,
            'deployments_today': 3
        },
        'market': {
            'volatility': 0.25,
            'volume_ratio': 2.5
        },
        'behavior': {
            'bounce_rate': 0.15,
            'new_user_ratio': 0.35
        }
    }
    
    hormone_impulses = converter.convert(content_batch)
    
    print("\nFull Pipeline Demo:")
    print("=" * 50)
    print("Content processed:")
    for content_type, content in content_batch.items():
        print(f"  {content_type}: {str(content)[:60]}...")
    
    print(f"\nResulting hormone impulses:")
    for hormone, strength in hormone_impulses.items():
        print(f"  {hormone}: {strength:.3f}")
    
    return hormone_impulses


if __name__ == "__main__":
    demo_text_analysis()
    print("\n" + "="*70 + "\n")
    demo_full_pipeline() 