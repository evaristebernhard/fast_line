# Literature map for the physical-balance / minimum-correction line

This note is a research map for the mathematical direction currently developed in
`fast_line`.  It is intentionally separated from `main.tex` and
`power_tdi_problem_framework.tex`.

The old mathematical thread evolved roughly as

[
\text{SAV predictor}
\to
\text{scalar physical-balance defect}
\to
\text{energy correction}
\to
\text{COI-compatible correction}
\to
\text{metric minimum correction}
\to
\text{generic physical-balance projection}.
]

The purpose of this literature audit is to determine which pieces are already standard
projection/relaxation theory and which pieces may still be worth developing.

---

## 1. Norton--McLaren--Quispel--Stern--Zanna (2015)

**Projection methods and discrete gradient methods for preserving first integrals of ODEs**

- Journal: *Discrete and Continuous Dynamical Systems A*, 35(5), 2079--2098.
- DOI: 10.3934/dcds.2015.35.2079
- Open preprint: https://arxiv.org/abs/1302.2713
- PDF: https://arxiv.org/pdf/1302.2713

### Why it matters

This is the first paper to read carefully before claiming anything about our generic
projection wrapper.

It proves, for one first integral, that broad classes of linear projection methods

1. are equivalent to classes of discrete-gradient methods;
2. are locally well-defined and locally unique for sufficiently small steps;
3. preserve the order of the underlying integrator;
4. allow substantial freedom in the projection direction;
5. can be analysed without a step restriction that collapses when approaching critical
   points of the invariant.

### Consequence for us

The following are **not** enough for novelty by themselves:

- arbitrary base one-step method + projection;
- order preservation after projection;
- local existence/uniqueness;
- freedom in projection direction;
- basic critical-point robustness for invariant-preserving projection.

Our recent `physical_balance_degeneracy.py` therefore cannot be sold merely as
"projection remains regular near critical points"; Norton et al. already treat a strong
version of that issue for first-integral manifolds.

### Possible distinction

Our balance target is not necessarily a fixed first-integral level set
(I(y_{n+1})=I(y_n)).  It may be a one-step **physical balance manifold**

[
\mathcal B_n(y_{n+1})
=
H(y_{n+1})-H(y_n)-Q_n(y_n,y_{n+1})=0,
]

where the target itself contains dissipative/work terms.  Whether this genuinely changes
the projection theory enough to yield new mathematics must be proved, not assumed.

---

## 2. Ketcheson (2019)

**Relaxation Runge--Kutta Methods: Conservation and Stability for Inner-Product Norms**

- Journal: *SIAM Journal on Numerical Analysis* 57(6), 2850--2870.
- DOI: 10.1137/19M1263662
- Open preprint: https://arxiv.org/abs/1905.09847
- PDF: https://arxiv.org/pdf/1905.09847

### Why it matters

A standard RK update is modified by one scalar relaxation parameter.  The modified
method can enforce conservation or monotonicity of an inner-product norm while
retaining high order and many properties of the base RK method.

The method can also be interpreted as a projection along the RK update direction.

### Consequence for us

The following are already well established ideas:

- "one scalar algebraic correction";
- "retain the order of a high-order base RK method";
- "enforce conservation/dissipation with tiny additional cost";
- "use an arbitrary/high-order base RK method and correct afterward".

So our mathematical paper must not claim novelty merely from the existence of a scalar
correction parameter.

### Structural difference worth studying

Relaxation changes the solution along the RK increment direction and, in its classical
interpretation, changes the effective time point from (t_n+h) to
(t_n+\gamma h).

Our current physical-balance projection is a **fixed-time state correction**:
the time node remains (t_{n+1}=t_n+h), and the direction is chosen from a metric
or constrained physical geometry rather than necessarily the RK increment.

That distinction is real, but projection literature already contains fixed-time methods,
so it is not sufficient alone.

---

## 3. Ranocha--Ketcheson (2020)

**Relaxation Runge--Kutta Methods for Hamiltonian Problems**

- Journal: *Journal of Scientific Computing* 84(1), 2020.
- DOI: 10.1007/s10915-020-01277-y
- Open preprint: https://arxiv.org/abs/2001.04826
- PDF: https://arxiv.org/pdf/2001.04826
- Reproducibility repository:
  https://github.com/ranocha/Hamiltonian-RRK-notebooks

### Why it matters

This paper specializes relaxation RK to Hamiltonian systems and studies long-time
behaviour.  Exact energy conservation is not presented merely as a formal property:
they investigate qualitative trajectories, error growth, and superconvergence for
special Hamiltonian classes.

### Consequence for us

If we return to a mathematical paper, we also need a statement stronger than
"energy residual is machine precision".

Good targets would be:

- a demonstrably smaller physically meaningful trajectory distortion;
- a mode/phase error theorem;
- a special superconvergence phenomenon;
- a balance-specific stability property.

---

## 4. Ranocha--Sayyari--Dalcin--Parsani--Ketcheson (2020)

**Relaxation Runge--Kutta Methods: Fully-Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier--Stokes Equations**

- Journal: *SIAM Journal on Scientific Computing*.
- DOI: 10.1137/19M1263480
- Open preprint: https://arxiv.org/abs/1905.09129
- PDF: https://arxiv.org/pdf/1905.09129

### Why it matters

This extends relaxation from inner-product norms to general convex functionals.
Conservation, dissipation, and entropy conditions can be imposed using one additional
scalar nonlinear solve.

### Consequence for us

"General convex energy + one scalar solve + dissipative inequality" is already covered.
Our distinction has to involve the **specific physical balance rate**, not just monotone
energy decay.

---

## 5. Ishii--Sato--Matsuo (2024)

**Affine-invariant projection methods for conservative integration of differential equations**

- Journal: *JSIAM Letters* 16 (2024), 49--52.
- DOI: 10.14495/jsiaml.16.49
- Article: https://www.jstage.jst.go.jp/article/jsiaml/16/0/16_49/_article
- PDF: https://www.jstage.jst.go.jp/article/jsiaml/16/0/16_49/_pdf

### Why it matters

Ordinary orthogonal projection depends on coordinates/metric.  This paper explicitly
asks which projection choices preserve affine invariance.

### Consequence for us

Our use of a metric (G), an (h)-scaled metric, or mass-weighted COI geometry is
not an innocent implementation detail.  If the method is to be a mathematical paper,
we must say what transformations the chosen metric respects.

This is directly relevant to the earlier
[
\|\Delta x\|_h^2
=
h^{-2}\Delta\delta^T M\Delta\delta
+
\Delta\omega^T M\Delta\omega
]
construction.

---

## 6. Najafian--Vermeire (2025)

**Quasi-Orthogonal Runge--Kutta Projection Methods**

- Journal: *Journal of Computational Physics* 530 (2025), 113917.
- DOI: 10.1016/j.jcp.2025.113917
- Open preprint: https://arxiv.org/abs/2409.18328
- PDF: https://arxiv.org/pdf/2409.18328

### Why it matters

This is very close to the direction we were drifting toward.

The method

- applies projection after explicit RK;
- handles a single non-convex invariant, dissipative systems, and multiple invariants;
- projects an orthogonal direction into the span of RK stage derivatives;
- preserves linear invariants;
- keeps the time step fixed;
- adds low cost;
- preserves the order of the base method.

### Consequence for us

"fixed-time", "linear-invariant-preserving", "low-cost", "optimal/orthogonal direction",
and "dissipative system" are not enough for novelty.

Our COI-compatible minimum correction has substantial conceptual overlap with this line.

A serious comparison should answer:

1. Is our direction mathematically distinct from their quasi-orthogonal stage-space
   direction?
2. Do we enforce an **exact physical rate** while they enforce an admissibility /
   invariant condition?
3. Does our second-order mechanical metric give a theorem that their Euclidean /
   RK-stage-space construction does not?

---

## 7. Cheng--Liu--Shen (2020)

**A new Lagrange Multiplier approach for gradient flows**

- Open preprint: https://arxiv.org/abs/1911.08336
- PDF: https://arxiv.org/pdf/1911.08336

### Why it matters

This paper was already a warning for our old EC--SAV story.

The Lagrange-multiplier approach is designed specifically to dissipate the
**original** energy, whereas standard SAV schemes dissipate a modified energy.
The price is a nonlinear scalar/algebraic solve.

### Consequence for us

"modified energy -> original energy" is not a new principle.

What was genuinely nice in our old SAV derivation was instead the exact defect identity
[
\varepsilon
=
U(\widetilde\delta)+C_0-\widetilde r^2
]
and the fact that, for the Swing predictor, the entire original physical-balance defect
collapsed to this single scalar.  That is a problem-specific structural identity rather
than a generic claim of original-energy stability.

---

## 8. Li--Li--Shang--Xu (2026)

**A unified framework of energy-stable splitting exponential integrators for damped Hamiltonian systems**

- Open preprint: https://arxiv.org/abs/2603.01709
- PDF: https://arxiv.org/pdf/2603.01709

### Why it matters

This is a very recent and very close paper.

For linearly perturbed/damped Hamiltonian systems it constructs

- SEISAV: modified-energy decay with only a one-dimensional linear algebraic solve;
- SEILM: original-energy decay using a Lagrange multiplier and one nonlinear algebraic
  equation;
- splitting/exponential treatment of damping and Hamiltonian subflows.

### Consequence for us

A paper whose only claim is
"linearly implicit SAV + scalar nonlinear correction -> original physical energy decay
for a damped Hamiltonian system" is now much too close to existing work.

If we keep the old SAV predictor, we need a sharper theorem or application-specific
structure.

---

## 9. Liu--Xue--Liu (2026)

**Arbitrarily high-order conformal invariants-preserving integrator for damped Hamiltonian ODEs**

- Journal: *Applied Mathematics Letters* 177 (2026), 109896.
- DOI: 10.1016/j.aml.2026.109896

### Why it matters

This paper explicitly points out that ordinary discrete-gradient methods may preserve
monotone decay but not the **correct physical dissipation rate**.  It constructs
projection-based schemes that exactly preserve the dissipation rates of conformal
invariants, with arbitrarily high order.

### Consequence for us

Even "exact dissipation rate, not merely energy monotonicity" now has direct prior art.

This makes it risky to formulate our mathematical contribution as simply
"exact physical dissipation balance".

A remaining possible distinction is that our one-step balance may be a general work
identity with topology/event changes, not necessarily a conformal invariant with a known
closed rate.  That distinction has to be formalized.

---

## 10. Tapley (2025 preprint)

**Explicit invariant-preserving integration of differential equations using homogeneous projection**

- arXiv: https://arxiv.org/abs/2511.02131
- PDF: https://arxiv.org/pdf/2511.02131

### Why it matters

This is another very close recent development.

It starts from an arbitrary base integrator and projects to an invariant manifold.  It
uses homogeneous symmetries to make projection explicit, extends to multiple invariants,
and discusses invariant modulation at a prescribed rate.  The paper also identifies
the minimum-norm vector field choice with an orthogonal projection.

### Consequence for us

The generic slogan

[
\text{arbitrary base integrator}
+
\text{minimum-norm projection}
+
\text{prescribed invariant rate}
]

is no longer safe as a novelty statement.

---

# Power-system bridge papers

## 11. Tzounas--Dassios--Milano (2022)

**Small-Signal Stability Analysis of Numerical Integration Methods**

- Journal: *IEEE Transactions on Power Systems* 37(6), 4796--4806.
- Preprint: https://arxiv.org/abs/2201.09529
- PDF: https://arxiv.org/pdf/2201.09529

This paper gives the matrix-pencil language for numerical mode distortion:
[
\widetilde s_k=h^{-1}\log \widetilde z_k,
\qquad
d_{s,k}=\widetilde s_k-s_k.
]

It is useful as a **diagnostic framework**, not as the closest mathematical competitor.

## 12. Huang--Sun (2022)

**A Study on Energy Preservability of Runge--Kutta Methods in Power System Simulation**

- IEEE PES General Meeting 2022.
- Public copy:
  https://curent.utk.edu/wp-content/uploads/2024/07/Kaiyang_Huang_UTK_Kai_1_R0.pdf

This is the cleanest bridge from mathematical energy error to power-system numerical
damping.

It derives energy change / numerical damping for Euler and RK4 on SMIB, then tests
the idea on Kundur's two-area system.

---

# What the old mathematical paper still has

After this audit, the old storyline should be split into **known machinery** and
**potentially distinctive structure**.

## Mostly known machinery

Do not sell these as main contributions:

- arbitrary base integrator followed by projection;
- one scalar nonlinear correction;
- retaining the order of the base method;
- orthogonal/minimum-norm projection;
- protecting linear invariants;
- original energy instead of modified SAV energy;
- monotone energy dissipation;
- exact dissipation rate in standard damped-Hamiltonian settings;
- generic robustness near critical points.

## Potentially worth keeping/reformulating

### A. Dynamic physical-balance manifold rather than a fixed invariant manifold

Our target can be
[
\mathcal B_n(y)
=
H(y)-H(y_n)-Q_n(y_n,y)=0,
]
where (Q_n) is a discrete physical work/dissipation functional and may depend on the
accepted endpoint.

This is different in form from projecting onto a fixed level set
(I(y)=I(y_n)).  It is also more general than a known multiplicative conformal law
(I_{n+1}=\alpha(h)I_n).

The mathematical question is whether this gives genuinely new existence/order/geometry
results, or whether it is merely a time-dependent invariant after state augmentation.

### B. Defect identity from an inexpensive predictor

For the old SAV-based Swing predictor,
[
\mathcal B_n(\widetilde y)=\varepsilon
=
U(\widetilde\delta)+C_0-\widetilde r^2.
]

This is stronger and more specific than saying "project the endpoint".
It identifies the complete physical-balance error with one auxiliary-variable defect and
explains why the correction is tiny.

### C. Second-order mechanical metric and turning-point scaling

The (h)-scaled metric
[
\|\Delta x\|_h^2
=
h^{-2}\Delta\delta^TM\Delta\delta
+
\Delta\omega^TM\Delta\omega
]
was introduced to respect the natural one-step scales
(
\Delta\delta=O(h^2)
)
and
(
\Delta\omega=O(h)
)
near turning points.

The resulting
[
\varepsilon=O(h^4),
\qquad
\Delta\delta=O(h^4)\text{ or smaller},
\qquad
\Delta\omega=O(h^3)
]
turning-point structure may still be worth isolating if it is not already a routine
consequence of general projection theory.

### D. Curvature-matched cheap projection

Our later path
[
\gamma(\lambda)
=
y^*+\lambda d+\lambda^2 k
]
was designed to agree with the true nonlinear metric-nearest projection to
(O(r^3)) while still solving one scalar equation.

This is pure numerical-analysis territory, but it may actually be a more defensible
mathematical question than the generic "physical-balance wrapper" if no direct prior
result exists.  It needs a dedicated literature search before any novelty claim.

### E. Power-system-specific geometry

COI/rotational symmetry gives a natural protected linear subspace.  Combining
physical-balance projection with that symmetry and then quantifying its effect on
electromechanical modes is likely stronger as an application-specific contribution than
as a universal projection theorem.

---

# Current recommendation

Do **not** throw away the old mathematical work, but stop treating the following as one
single paper.

There are now two possible papers:

1. **Numerical-analysis paper**:
   focus narrowly on a genuinely new mathematical object, e.g. dynamic balance
   manifolds, curvature-matched metric projection, or a special turning-point theorem.
   Power systems can be one example, not the identity of the method.

2. **Power-system paper**:
   use only the minimum amount of projection theory needed, and focus on what exact
   physical balance changes in rotor-angle / damping / CCT calculations.

The next step is to compare our exact formulas line-by-line against papers 1, 6, 8, 9,
and 10 before choosing which of A--D survives as a defensible theorem.
