import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def get_cdf(cdf_name: str, loc=0, scale=1):
    """Return a target cdf distribution."""
    tfd = tfp.distributions

    if cdf_name == "gumbel":
        target_cdf = tfd.Gumbel(loc=loc, scale=scale)
    elif cdf_name == "bimix_gaussian":
        mix = 1/3
        bimix_gaussian = tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.-mix]),
            components=[
                tfd.Normal(loc=loc, scale=scale),
                tfd.Normal(loc=-loc, scale=scale),
        ])
        target_cdf = bimix_gaussian
    elif cdf_name == "exponomial":
        # The exponomial distribution is a negative gumbel distribution. 
        # Create a new distribution by inheriting the Gumbel class and overriding the cdf, prob, and sample methods.
        class Exponomial(tfd.Exponential):
            def cdf(self, x):
                return 1 - super(Exponomial, self).cdf(-x)
            def prob(self, x):
                return super(Exponomial, self).prob(-x)
            def sample(self, n=None, size=None):
                if n is not None:
                    return - super(Exponomial, self).sample(n)
                elif size is not None:
                    return - super(Exponomial, self).sample(size=size)
                else:
                    raise ValueError
            def mean(self):
                return - super(Exponomial, self).mean()
            def variance(self):
                return super(Exponomial, self).variance()
        target_cdf = Exponomial(rate=scale, force_probs_to_zero_outside_support=True)
    elif cdf_name == "gaussian":
        target_cdf = tfd.Normal(loc=loc, scale=scale)
    elif cdf_name == "beta":
        class Beta(tfd.Beta):
            def __init__(     
                self,
                concentration1,
                concentration0,
                validate_args=False,
                allow_nan_stats=True,
                force_probs_to_zero_outside_support=False,
                name='Beta',
                scale=1,
            ):
                super(Beta, self).__init__(
                    concentration0=concentration0,
                    concentration1=concentration1,
                    validate_args=validate_args,
                    allow_nan_stats=allow_nan_stats,
                    force_probs_to_zero_outside_support=force_probs_to_zero_outside_support,
                    name=name,
                )

                self.scale = scale

            def cdf(self, x):
                return super(Beta, self).cdf(x / self.scale)
            def prob(self, x):
                return super(Beta, self).prob(x /  self.scale)
            def sample(self, n=None, size=None):
                if n is not None:
                    return super(Beta, self).sample(n) * self.scale
                elif size is not None:
                    return super(Beta, self).sample(size=size) * self.scale
                else:
                    raise ValueError
            def mean(self):
                return super(Beta, self).mean() * self.scale
            def variance(self):
                return super(Beta, self).variance() * (self.scale**2)
        target_cdf = Beta(concentration0=0.5, concentration1=0.5, scale=scale, force_probs_to_zero_outside_support=True)
    else:
        raise ValueError("Invalid cdf_name")
    
    return target_cdf