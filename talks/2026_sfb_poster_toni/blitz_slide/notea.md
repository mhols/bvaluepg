# Suggested 1–2 minute pitch

not sure

Here is the declustered Italian earthquake catalogue. We observe noisy event
counts in spatial cells, and we want to infer the underlying intensity field
and its uncertainty.

Our spatial Poisson model links the rate to a latent Gaussian field through a
sigmoid function. That makes the posterior for the field non-Gaussian, so
sampling it directly is difficult.

We introduce auxiliary Poisson counts and Pólya–Gamma variables. Conditional
on them, the field update becomes Gaussian. Its precision is the sparse prior
precision Q plus a diagonal matrix. We can therefore keep the sparse
computational structure while sampling the posterior.

On the poster, I show synthetic checks of the sampler and what the posterior
reveals for the Italian catalogue.
