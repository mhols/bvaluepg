'''
nur mal idee

'''
def target_pdf(x):
    # z.B. eine komplizierte Dichte
    return ...

def proposal_pdf(x):
    # einfache Dichte, z. B. Normalverteilung
    return ...

def proposal_sampler():
    return np.random.normal()

M = ...  # Konstante, so dass target_pdf(x) <= M * proposal_pdf(x)

samples = []
while len(samples) < 1000:
    x = proposal_sampler()
    u = np.random.uniform(0, 1)
    if u < target_pdf(x) / (M * proposal_pdf(x)):
        samples.append(x)