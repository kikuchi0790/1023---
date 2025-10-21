"""
Test cases for Bayesian Inference Analysis
"""

import pytest
import numpy as np
from utils.bayesian_analysis import BayesianAnalyzer


class TestBayesianAnalyzer:
    
    def test_initialization(self):
        matrix = np.array([
            [0, 3, 0],
            [2, 0, 4],
            [0, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = BayesianAnalyzer(matrix, nodes, n_bootstrap=100)
        
        assert analyzer.n == 3
        assert len(analyzer.node_names) == 3
        assert analyzer.n_edges == 4
        assert analyzer.n_bootstrap == 100
        assert analyzer.credible_level == 0.95
        assert 'mu_0' in analyzer.prior_params
    
    def test_prior_initialization(self):
        matrix = np.array([[0, 5], [3, 0]])
        nodes = ["A", "B"]
        
        analyzer_weak = BayesianAnalyzer(matrix, nodes, prior_type='weak_informative')
        assert analyzer_weak.prior_params['sigma_0'] == 5.0
        
        analyzer_uninf = BayesianAnalyzer(matrix, nodes, prior_type='uninformative')
        assert analyzer_uninf.prior_params['sigma_0'] == 100.0
    
    def test_bootstrap_approximation(self):
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = BayesianAnalyzer(matrix, nodes, n_bootstrap=100)
        bootstrap_samples = analyzer._bootstrap_approximation()
        
        assert len(bootstrap_samples) == analyzer.n_edges
        
        for edge, samples in bootstrap_samples.items():
            assert isinstance(samples, np.ndarray)
            assert len(samples) == 100
            assert isinstance(edge, tuple)
            assert len(edge) == 2
    
    def test_posterior_distribution(self):
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = BayesianAnalyzer(matrix, nodes, n_bootstrap=100)
        bootstrap_samples = analyzer._bootstrap_approximation()
        posterior_mean, posterior_std = analyzer._compute_posterior_distribution(bootstrap_samples)
        
        assert len(posterior_mean) == analyzer.n_edges
        assert len(posterior_std) == analyzer.n_edges
        
        for edge in analyzer.edges:
            assert edge in posterior_mean
            assert edge in posterior_std
            assert posterior_std[edge] >= 0
    
    def test_credible_intervals(self):
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = BayesianAnalyzer(matrix, nodes, n_bootstrap=100, credible_level=0.95)
        bootstrap_samples = analyzer._bootstrap_approximation()
        credible_intervals = analyzer._compute_credible_intervals(bootstrap_samples)
        
        assert len(credible_intervals) == analyzer.n_edges
        
        for edge, (mean_val, lower, upper) in credible_intervals.items():
            assert lower <= mean_val <= upper or abs(mean_val - lower) < 1e-6 or abs(mean_val - upper) < 1e-6
            assert upper >= lower
    
    def test_uncertainty_scores(self):
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = BayesianAnalyzer(matrix, nodes, n_bootstrap=100)
        bootstrap_samples = analyzer._bootstrap_approximation()
        credible_intervals = analyzer._compute_credible_intervals(bootstrap_samples)
        posterior_mean, _ = analyzer._compute_posterior_distribution(bootstrap_samples)
        
        uncertainty_scores = analyzer._compute_uncertainty_scores(credible_intervals, posterior_mean)
        
        assert len(uncertainty_scores) == analyzer.n_edges
        
        for edge, score in uncertainty_scores.items():
            assert score >= 0
    
    def test_high_uncertainty_identification(self):
        uncertainty_scores = {
            ("A", "B"): 0.8,
            ("B", "C"): 0.2,
            ("C", "A"): 0.6,
            ("A", "C"): 0.9
        }
        
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        analyzer = BayesianAnalyzer(matrix, nodes)
        
        high_uncertainty_edges = analyzer._identify_high_uncertainty_edges(uncertainty_scores, top_k=2)
        
        assert len(high_uncertainty_edges) == 2
        assert high_uncertainty_edges[0][2] == 0.9
        assert high_uncertainty_edges[1][2] == 0.8
    
    def test_comprehensive_analysis(self):
        matrix = np.array([
            [0, 5, 3, 0],
            [2, 0, 4, 1],
            [1, 3, 0, 2],
            [0, 0, 1, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = BayesianAnalyzer(matrix, nodes, n_bootstrap=200)
        result = analyzer.compute_bayesian_inference()
        
        assert isinstance(result.posterior_mean, dict)
        assert isinstance(result.posterior_std, dict)
        assert isinstance(result.credible_intervals, dict)
        assert isinstance(result.high_uncertainty_edges, list)
        assert isinstance(result.uncertainty_scores, dict)
        assert isinstance(result.interpretation, str)
        assert result.computation_time > 0
        
        assert result.n_edges > 0
        assert result.n_bootstrap == 200
        assert result.credible_level == 0.95
        
        assert len(result.posterior_mean) == result.n_edges
        assert len(result.credible_intervals) == result.n_edges
    
    def test_empty_graph(self):
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        
        analyzer = BayesianAnalyzer(matrix, nodes)
        result = analyzer.compute_bayesian_inference()
        
        assert result is not None
        assert result.n_edges == 0
        assert len(result.posterior_mean) == 0
        assert len(result.high_uncertainty_edges) == 0
    
    def test_different_credible_levels(self):
        matrix = np.array([[0, 5], [3, 0]])
        nodes = ["A", "B"]
        
        analyzer_90 = BayesianAnalyzer(matrix, nodes, n_bootstrap=100, credible_level=0.90)
        result_90 = analyzer_90.compute_bayesian_inference()
        
        analyzer_99 = BayesianAnalyzer(matrix, nodes, n_bootstrap=100, credible_level=0.99)
        result_99 = analyzer_99.compute_bayesian_inference()
        
        assert result_90.credible_level == 0.90
        assert result_99.credible_level == 0.99
        
        edge = ("A", "B")
        ci_90 = result_90.credible_intervals[edge]
        ci_99 = result_99.credible_intervals[edge]
        
        width_90 = ci_90[2] - ci_90[1]
        width_99 = ci_99[2] - ci_99[1]
        
        assert width_99 >= width_90 or abs(width_99 - width_90) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
