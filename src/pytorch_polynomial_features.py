import math
import torch
import numpy as np
from scipy.special import comb, perm
from itertools import combinations_with_replacement, combinations

class PolynomialFeatures(torch.nn.Module):
    """
    PyTorch implementation of polynomial features that supports backpropagation.
    Drop-in replacement for sklearn.preprocessing.PolynomialFeatures.

    X is of shape(n_samples, n_features_in)
    """
    def __init__(self,
                 degree: int,
                 interaction_only: bool = False,
                 include_bias: bool = True):
        super(PolynomialFeatures, self).__init__()

        if degree < 1:
            raise ValueError("Degree must be at least 1")

        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def transform(self, x):
        """
        Transform data to polynomial features.

        ## Parameters
        - **x**: torch.Tensor of shape (n_samples, n_features_in)

        ## Returns
        - **output**: torch.Tensor of shape (n_samples, n_output_features_)
        """
        comb_f = combinations if self.interaction_only else combinations_with_replacement

        # Create combinations
        output = []
        for d in range(1, self.degree + 1):
            for combo in comb_f(range(self.n_feature_in), d):
                output.append(torch.prod(x[:, combo], dim=1))
        output = torch.stack(output, dim=1)

        # Add bias
        if self.include_bias:
            bias = torch.ones(x.shape[0], 1, device=x.device)
            output = torch.cat([bias, output], dim=1)

        return output

    def _combo_to_dict(self, combo):
        """
        Convert a combination of features to a dictionary of feature names to powers.  
        Example: If combo is (0, 0, 1, 3), then the output is {0: 2, 1: 1, 3: 1}  

        ## Parameters
        - **combo**: list of ints

        ## Returns
        - **dict**: dict of feature names and powers

        ## Notes
        - This is used in `get_feature_names_out(...)` as a helper function.
        """
        # Initialize an empty dictionary to store the counts
        count_dict = {}

        # Iterate over each element in the tuple
        for num in combo:
            count_dict[num] = count_dict.get(num, 0) + 1

        return count_dict

    def get_feature_names_out(self):
        """
        Get the feature names of the polynomial features.  
        Must be called after `fit(...)`.  

        ## Returns
        - **output**: Numpy array of feature names (`object` dtype).
        """
        comb_f = combinations if self.interaction_only else combinations_with_replacement
        feature_names = [f"x{i}" for i in range(self.n_feature_in)]

        # Create combinations
        output = []

        if self.include_bias:
            output.append("1")

        for d in range(1, self.degree + 1):
            for combo in comb_f(range(self.n_feature_in), d):
                combo_dict = self._combo_to_dict(combo)
                output.append(" ".join([f"{feature_names[idx]}^{power}" if power > 1 else feature_names[idx] for idx, power in combo_dict.items()]))

        output = np.asarray(output, dtype=object)

        return output

    def fit(self, X):
        """
        Fit the polynomial features to the data. Just stores the number of input and output features.  

        ## Parameters
        - **X**: torch.Tensor of shape (n_samples, n_features_in)

        ## Returns
        - **self**: self
        """
        self.n_feature_in = X.shape[-1]

        self.n_output_features_ = int(sum([comb(self.n_feature_in , d, repetition=(not self.interaction_only)) for d in range(1, self.degree + 1)]))

        if self.include_bias:
            self.n_output_features_ += 1

        return self

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        ## Parameters
        - **X**: torch.Tensor of shape (n_samples, n_features_in)

        ## Returns
        - **output**: torch.Tensor of shape (n_samples, n_output_features_)

        ## Notes
        - This is a convenience function that calls `fit(...)` and `transform(...)` in sequence.
        """
        self.fit(X)
        return self.transform(X)

def test_polynomial_features():
    """Test function to verify the implementation works correctly."""
    from sklearn.preprocessing import PolynomialFeatures as SklearnPolyFeatures

    print("Testing PyTorch PolynomialFeatures...")
    
    # Create test data
    X = torch.randn(10, 3, requires_grad=True)
    
    for degree in [1, 2, 3]:
        for interaction_only in [True, False]:
            for include_bias in [True, False]:
                print(f"Degree: {degree}, Interaction only: {interaction_only}, Include bias: {include_bias}")

                # Test polynomial features
                pf = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only)
                X_poly = pf.fit_transform(X)
                
                # Test backpropagation
                loss = X_poly.sum()
                loss.backward()
                
                print(f"Gradients computed successfully: {X.grad is not None}")
                print(f"Gradient shape: {X.grad.shape}")
                
                sklearn_pf = SklearnPolyFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only)
                X_sklearn = sklearn_pf.fit_transform(X.detach().numpy())
                
                # Check if shapes match
                print(f"Shape matches sklearn: {X_poly.shape[1] == X_sklearn.shape[1]}")
                
                # Check if values are close (they should be identical)
                diff = torch.abs(X_poly.detach() - torch.tensor(X_sklearn)).max()
                print(f"Max difference with sklearn: {diff.item():.2e}")

                # Print library
                print("Custom terms: ", pf.get_feature_names_out())
                print("Sklearn terms:", sklearn_pf.get_feature_names_out())

                # Print term shape
                print("Custom terms:", pf.n_output_features_)
                print("Sklearn terms:", sklearn_pf.n_output_features_)

                if pf.n_output_features_ != sklearn_pf.n_output_features_:
                    raise ValueError("Custom and sklearn polynomial features do not match")

                if diff.item() > 1e-6:
                    raise ValueError("Custom and sklearn polynomial features do not match")
                print()
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_polynomial_features()
