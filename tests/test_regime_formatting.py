"""
Test suite for regime-based title formatting.

Tests cover:
1. Identifiable Regime formatting (l1_error < 0.1)
2. Partial Identifiability formatting (0.1 <= l1_error < 0.3)
3. Confusable Regime formatting (l1_error >= 0.3)
"""

import unittest


def format_main_title(l1_error: float, final_acc: float) -> str:
    """
    Format main title based on recovery quality (L1 error).
    
    This function mirrors the logic from make_figures_recovery.py
    for determining the regime based on l1_error values.
    
    Parameters
    ----------
    l1_error : float
        L1 distance between true and learned weights.
    final_acc : float
        Final training accuracy.
    
    Returns
    -------
    main_title : str
        Formatted title string indicating the regime.
    """
    if l1_error < 0.1:
        regime = "Identifiable Regime"
    elif l1_error < 0.3:
        regime = "Partial Identifiability"
    else:
        regime = "Confusable Regime"
    
    return f"{regime} (L1={l1_error:.3f}, acc={final_acc:.2f})"


class TestRegimeFormatting(unittest.TestCase):
    """Test suite for regime-based title formatting."""
    
    def test_identifiable_regime_lower_bound(self):
        """Test that l1_error=0.0 is correctly formatted as Identifiable Regime."""
        l1_error = 0.0
        final_acc = 0.95
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Identifiable Regime", result)
        self.assertIn("L1=0.000", result)
        self.assertIn("acc=0.95", result)
    
    def test_identifiable_regime_upper_bound(self):
        """Test that l1_error=0.09 is correctly formatted as Identifiable Regime."""
        l1_error = 0.09
        final_acc = 0.88
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Identifiable Regime", result)
        self.assertIn("L1=0.090", result)
        self.assertIn("acc=0.88", result)
    
    def test_identifiable_regime_typical_value(self):
        """Test that a typical low l1_error is correctly formatted as Identifiable Regime."""
        l1_error = 0.05
        final_acc = 0.92
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Identifiable Regime", result)
        self.assertIn("L1=0.050", result)
        self.assertIn("acc=0.92", result)
    
    def test_partial_identifiability_lower_bound(self):
        """Test that l1_error=0.1 is correctly formatted as Partial Identifiability."""
        l1_error = 0.1
        final_acc = 0.80
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Partial Identifiability", result)
        self.assertIn("L1=0.100", result)
        self.assertIn("acc=0.80", result)
    
    def test_partial_identifiability_upper_bound(self):
        """Test that l1_error=0.29 is correctly formatted as Partial Identifiability."""
        l1_error = 0.29
        final_acc = 0.75
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Partial Identifiability", result)
        self.assertIn("L1=0.290", result)
        self.assertIn("acc=0.75", result)
    
    def test_partial_identifiability_typical_value(self):
        """Test that a typical mid-range l1_error is correctly formatted as Partial Identifiability."""
        l1_error = 0.2
        final_acc = 0.78
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Partial Identifiability", result)
        self.assertIn("L1=0.200", result)
        self.assertIn("acc=0.78", result)
    
    def test_confusable_regime_lower_bound(self):
        """Test that l1_error=0.3 is correctly formatted as Confusable Regime."""
        l1_error = 0.3
        final_acc = 0.70
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Confusable Regime", result)
        self.assertIn("L1=0.300", result)
        self.assertIn("acc=0.70", result)
    
    def test_confusable_regime_high_value(self):
        """Test that a high l1_error is correctly formatted as Confusable Regime."""
        l1_error = 0.5
        final_acc = 0.65
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Confusable Regime", result)
        self.assertIn("L1=0.500", result)
        self.assertIn("acc=0.65", result)
    
    def test_confusable_regime_very_high_value(self):
        """Test that a very high l1_error is correctly formatted as Confusable Regime."""
        l1_error = 1.0
        final_acc = 0.55
        result = format_main_title(l1_error, final_acc)
        
        self.assertIn("Confusable Regime", result)
        self.assertIn("L1=1.000", result)
        self.assertIn("acc=0.55", result)
    
    def test_formatting_precision(self):
        """Test that formatting uses correct decimal precision."""
        l1_error = 0.12345
        final_acc = 0.876543
        result = format_main_title(l1_error, final_acc)
        
        # L1 should be formatted to 3 decimal places
        self.assertIn("L1=0.123", result)
        # Accuracy should be formatted to 2 decimal places
        self.assertIn("acc=0.88", result)
    
    def test_regime_boundaries_exclusive(self):
        """Test that regime boundaries are correctly exclusive/inclusive."""
        # Just below 0.1 threshold - should be Identifiable
        result_below = format_main_title(0.099, 0.9)
        self.assertIn("Identifiable Regime", result_below)
        
        # At 0.1 threshold - should be Partial Identifiability
        result_at = format_main_title(0.1, 0.9)
        self.assertIn("Partial Identifiability", result_at)
        
        # Just below 0.3 threshold - should be Partial Identifiability
        result_below_upper = format_main_title(0.299, 0.75)
        self.assertIn("Partial Identifiability", result_below_upper)
        
        # At 0.3 threshold - should be Confusable
        result_at_upper = format_main_title(0.3, 0.75)
        self.assertIn("Confusable Regime", result_at_upper)


if __name__ == '__main__':
    unittest.main()
