"""Feature maps phi for linear attention.

The whole research question lives here: linear attention replaces
softmax(q . k) with <phi(q), phi(k)>. Performer/FAVOR+ uses positive random
features whose projection rows are sampled from a Gaussian (optionally
orthogonalised) to approximate the *softmax* kernel. Our thesis is that a
projection sampler derived from a *quantum kernel* yields a better inductive
bias at matched feature dimension r.

So every feature map below shares the exact same FAVOR+ positive-feature
machinery and differs ONLY in how the projection matrix W (r x d) is sampled.
That makes the central comparison (M1: quantum sampler vs gaussian sampler at
matched r) fair by construction -- same code path, one knob.

    phi(x) = exp(-||x||^2 / 2) / sqrt(r) * exp(W x)     (positive variant)
    E[ phi(x) . phi(y) ] ~ exp(x . y)                   (the softmax kernel)

References:
    Choromanski et al., "Rethinking Attention with Performers", ICLR 2021
        arXiv:2009.14794
    Rahimi & Recht, "Random Features for Large-Scale Kernel Machines", 2007
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import nn


def _orthogonal_gaussian(rows: int, cols: int, device=None, dtype=None) -> torch.Tensor:
    """Sample an (rows x cols) matrix with (near-)orthogonal rows and Gaussian
    row norms -- the FAVOR+ "orthogonal random features" construction. Blocks of
    `cols` orthonormal rows are stacked, then each row is rescaled to a chi-
    distributed norm so the marginal stays N(0, I)."""
    blocks = []
    remaining = rows
    while remaining > 0:
        n = min(remaining, cols)
        g = torch.randn(cols, cols, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(g)            # q: (cols x cols), orthonormal rows
        blocks.append(q[:n])
        remaining -= n
    q = torch.cat(blocks, dim=0)             # (rows x cols)
    # rescale rows to chi-distributed norms (matches the Gaussian marginal)
    norms = torch.randn(rows, cols, device=device, dtype=dtype).norm(dim=1)
    return q * norms.unsqueeze(1)


class FeatureMap(nn.Module, ABC):
    """Maps q/k of shape (b, h, n, d) -> positive features (b, h, n, r)."""

    def __init__(self, dim_head: int, num_features: int):
        super().__init__()
        self.dim_head = dim_head
        self.num_features = num_features

    @abstractmethod
    def projection(self, device, dtype) -> torch.Tensor:
        """Return the projection matrix W of shape (num_features, dim_head)."""

    def forward(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        # x: (b, h, n, d). Stabilised FAVOR+ positive features.
        w = self.projection(x.device, x.dtype)              # (r, d)
        x = x * (self.dim_head ** -0.25)                    # softmax-kernel scaling
        proj = torch.einsum('b h n d, r d -> b h n r', x, w)
        diag = (x ** 2).sum(dim=-1, keepdim=True) * 0.5     # ||x||^2 / 2
        # subtract a stabiliser (max over features) to avoid exp overflow
        if is_query:
            stab = proj.amax(dim=-1, keepdim=True)
        else:
            stab = proj.amax(dim=(-2, -1), keepdim=True)
        feats = torch.exp(proj - diag - stab) + 1e-6
        return feats / math.sqrt(self.num_features)


class GaussianRFF(FeatureMap):
    """Baseline #4 -- generic random features (Gaussian / orthogonal sampler).

    This is the control that isolates the contribution of the *quantum* kernel:
    if our quantum sampler cannot beat this at matched r, the thesis fails (M1).
    """

    def __init__(self, dim_head: int, num_features: int, orthogonal: bool = True,
                 redraw: bool = False):
        super().__init__(dim_head, num_features)
        self.orthogonal = orthogonal
        self.redraw = redraw                                # resample W every fwd?
        self.register_buffer('_w', self._sample(), persistent=True)

    def _sample(self) -> torch.Tensor:
        if self.orthogonal:
            return _orthogonal_gaussian(self.num_features, self.dim_head)
        return torch.randn(self.num_features, self.dim_head)

    def projection(self, device, dtype) -> torch.Tensor:
        if self.redraw and self.training:
            return self._sample().to(device=device, dtype=dtype)
        return self._w.to(device=device, dtype=dtype)


class ShiftInvariantRFF(FeatureMap):
    """Bochner random-Fourier-features base for a *shift-invariant* kernel.

    By Bochner's theorem every continuous shift-invariant PD kernel is the
    Fourier transform of a non-negative spectral measure p(omega):

        k(x - y) = E_{omega ~ p, b ~ U[0, 2pi]} [ 2 cos(omega.x + b) cos(omega.y + b) ]
                 = Integral cos(omega.(x - y)) p(omega) d_omega.

    So phi(x) = sqrt(2/r) cos(W x + b), with the r rows of W drawn from p and
    b ~ U[0, 2pi], gives an unbiased estimate  <phi(x), phi(y)> -> k(x - y).

    Subclasses differ ONLY in the spectral sampler `_sample_omega` -- that is
    the one knob the whole M1 comparison turns on. Gaussian RFF samples an
    isotropic continuous Gaussian (-> RBF kernel); the quantum map samples a
    discrete integer-harmonic lattice (-> the angle-embedding quantum kernel).

    cos features are not non-negative (Bochner forces this for a shift-invariant
    kernel), so two stabilisers make them usable inside normalised linear
    attention WITHOUT changing the kernel they estimate:

    * `bandwidth` defaults to dim_head**-0.5. A product kernel over d dims
      concentrates to ~2^-d for any bandwidth too large -- the quantum-kernel
      "exponential concentration" failure (Thanasilp et al. 2022; Shaydulin &
      Wild 2022; Canatar et al. 2023). The 1/sqrt(d) default keeps the kernel
      off the floor; treat it as a tuned hyperparameter (same budget for the
      Gaussian control -- fairness).
    * `dc` appends a single non-negative constant feature sqrt(dc) at attention
      time, so the denominator phi(q).sum_k phi(k) gets a +dc*N floor and stays
      positive (a mild uniform-attention prior, identical for both cos variants
      -> still fair). The kernel-fidelity study (Table 4) uses dc=0, so the
      scientific phi.phi^T -> K claim is on the bare features.
    """

    def __init__(self, dim_head: int, num_features: int,
                 bandwidth: float | None = None, dc: float = 0.0,
                 qk_norm: bool = False):
        super().__init__(dim_head, num_features)
        # QK-norm: L2-normalise q,k onto the unit sphere before projecting
        # [ViT-22B arXiv:2302.05442; Henry et al. arXiv:2010.04245]. Bounds the
        # kernel (no concentration / winner-take-all) and, for the isotropic
        # Gaussian control, turns the RBF kernel into a dot-product kernel
        # f(<q,k>) since ||q-k||^2 = 2 - 2<q,k> on the sphere. With ||q||=1 the
        # projection w.q has scale ~bandwidth INDEPENDENT of d, so the sensible
        # auto-bandwidth is O(1) rather than 1/sqrt(d).
        self.qk_norm = qk_norm
        if bandwidth is None:
            bandwidth = 1.0 if qk_norm else dim_head ** -0.5
        self.bandwidth = bandwidth
        self.dc = dc
        self.register_buffer('_w', self._sample_omega(), persistent=True)   # (r, d)
        self.register_buffer('_b', 2 * math.pi * torch.rand(num_features),  # (r,)
                             persistent=True)

    @abstractmethod
    def _sample_omega(self) -> torch.Tensor:
        """Draw the (num_features x dim_head) frequency matrix from p(omega)."""

    def projection(self, device, dtype) -> torch.Tensor:
        return self._w.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        # x: (b, h, n, d) -> cos features (b, h, n, r [+1]). is_query unused (the
        # estimator is symmetric in q/k for a shift-invariant kernel).
        if self.qk_norm:                                            # onto unit sphere
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w = self.projection(x.device, x.dtype)                      # (r, d)
        b = self._b.to(device=x.device, dtype=x.dtype)              # (r,)
        proj = torch.einsum('b h n d, r d -> b h n r', x, w) + b
        feats = math.sqrt(2.0 / self.num_features) * torch.cos(proj)
        if self.dc > 0:                                             # +ve denom floor
            pad = feats.new_full((*feats.shape[:-1], 1), math.sqrt(self.dc))
            feats = torch.cat([feats, pad], dim=-1)
        return feats


class GaussianRFFBochner(ShiftInvariantRFF):
    """Classical Gaussian RFF (Rahimi & Recht 2007) -- the M1 control.

    Spectral measure p(omega) = N(0, bandwidth^2 I) (continuous, isotropic) so
    <phi(x), phi(y)> -> exp(-bandwidth^2 ||x - y||^2 / 2), the RBF kernel. This
    shares the EXACT cos-feature code path with QuantumKernelFeatureMap; the
    only difference is the spectrum. If the quantum spectrum cannot beat this
    isotropic-Gaussian spectrum at matched r, the central thesis fails (M1).
    """

    def _sample_omega(self) -> torch.Tensor:
        return self.bandwidth * torch.randn(self.num_features, self.dim_head)


class QuantumKernelFeatureMap(ShiftInvariantRFF):
    """Angle/product quantum-kernel map (the depth-L map studied in the paper) --
    random features of a real quantum-embedding kernel.

    Embedding: each input coordinate x_j is angle-encoded as RZ(bandwidth*x_j)
    on |+>, with `layers` (L) such qubits per feature, as a product state. The
    exact kernel is closed-form and shift-invariant:

        K(x, x') = prod_j [ (1 + cos(bandwidth*(x_j - x'_j))) / 2 ] ** L .

    Per coordinate, (1 + cos t)/2 = 1/2 + 1/4 e^{it} + 1/4 e^{-it}, so the
    spectral measure is the DISCRETE integer-harmonic lattice: omega_j is
    bandwidth times a sum of L iid draws from {0:1/2, +1:1/4, -1:1/4}. Drawing
    W this way and using cos features reproduces K exactly in expectation --
    verified against an exact statevector simulation in `quantum_kernel_matrix`
    / `quantum_kernel_fidelity` (Table 4). No quantum hardware, no faking: the
    quantum kernel only sets WHICH frequencies we sample.

    The discrete lattice (vs the continuous Gaussian of GaussianRFFBochner) is
    the inductive-bias difference the paper tests; `layers` is the qubit-depth
    knob of ablation Table 3c.
    """

    def __init__(self, dim_head: int, num_features: int,
                 bandwidth: float | None = None, layers: int = 1, dc: float = 0.0,
                 qk_norm: bool = False):
        self.layers = layers
        super().__init__(dim_head, num_features, bandwidth=bandwidth, dc=dc,
                         qk_norm=qk_norm)

    def _sample_omega(self) -> torch.Tensor:
        # each coord = sum of L iid {0:1/2, +1:1/4, -1:1/4}, scaled by bandwidth
        steps = torch.multinomial(
            torch.tensor([0.5, 0.25, 0.25]),
            self.num_features * self.dim_head * self.layers,
            replacement=True,
        ).view(self.num_features, self.dim_head, self.layers)
        vals = torch.tensor([0.0, 1.0, -1.0])[steps]                # {0,+1,-1}
        return self.bandwidth * vals.sum(dim=-1)                    # (r, d)


class IQPKernelFeatureMap(FeatureMap):
    """Entangling IQP quantum-kernel map (the coupling-c map studied in the paper) --
    random features of a ZZ/IQP quantum kernel.

    Embedding |psi(x)> = U_Z(x) H^n |0>, with U_Z diagonal: on basis state
    s in {+1,-1}^d the phase is
        phi_s(x) = bw*(s.x) + (coupling/2)*bw^2*[(s.x)^2 - ||x||^2],
    where the ZZ-coupling term sum_{j<k} x_j x_k s_j s_k collapses to
    (1/2)[(s.x)^2 - ||x||^2] because s_j^2 = 1. The quantum kernel
        K(x,x') = |<psi(x)|psi(x')>|^2 = E_{s,s' ~ {+-1}^d}[ cos(g(x) - g(x')) ],
        g(x) = h(bw*s.x) - h(bw*s'.x),   h(t) = t + (coupling/2) t^2,
    so the random feature is phi(x) = sqrt(2/r) cos(h(bw*s.x) - h(bw*s'.x) + b)
    with s, s' ~ {+-1}^d and b ~ U[0,2pi]. Two sign-projections per feature ->
    O(d*r); the O(d^2) ZZ kernel is NEVER built.

    The quadratic h-term is the ENTANGLEMENT. `coupling`=0 removes the ZZ gates
    (a non-entangled control on the identical code path) -- the one-knob ablation
    that isolates whether entanglement is the active ingredient. Verified against
    an exact statevector sim in `iqp_kernel_matrix` / `iqp_kernel_fidelity`.

    Unlike GaussianRFFBochner / QuantumKernelFeatureMap (separable / shift-
    invariant), this kernel carries genuine cross-coordinate (pairwise) structure
    that no isotropic random feature can represent -- the make-or-break wedge.
    """

    def __init__(self, dim_head: int, num_features: int,
                 bandwidth: float | None = None, coupling: float = 1.0,
                 dc: float = 0.0, qk_norm: bool = False):
        super().__init__(dim_head, num_features)
        self.qk_norm = qk_norm
        self.coupling = coupling
        self.dc = dc
        if bandwidth is None:
            bandwidth = 1.0 if qk_norm else dim_head ** -0.5
        self.bandwidth = bandwidth
        # two random computational-basis sign patterns per feature (r, d)
        self.register_buffer('_s', torch.randint(0, 2, (num_features, dim_head),
                             dtype=torch.float32) * 2 - 1, persistent=True)
        self.register_buffer('_sp', torch.randint(0, 2, (num_features, dim_head),
                             dtype=torch.float32) * 2 - 1, persistent=True)
        self.register_buffer('_b', 2 * math.pi * torch.rand(num_features),
                             persistent=True)

    def projection(self, device, dtype) -> torch.Tensor:  # interface only
        return self._s.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        if self.qk_norm:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        x = x * self.bandwidth
        s = self._s.to(device=x.device, dtype=x.dtype)
        sp = self._sp.to(device=x.device, dtype=x.dtype)
        b = self._b.to(device=x.device, dtype=x.dtype)
        u = torch.einsum('b h n d, r d -> b h n r', x, s)
        up = torch.einsum('b h n d, r d -> b h n r', x, sp)
        g = (u - up) + 0.5 * self.coupling * (u * u - up * up)      # h(u)-h(u')
        feats = math.sqrt(2.0 / self.num_features) * torch.cos(g + b)
        if self.dc > 0:                                            # +ve denom floor
            pad = feats.new_full((*feats.shape[:-1], 1), math.sqrt(self.dc))
            feats = torch.cat([feats, pad], dim=-1)
        return feats


class QPSANFeatureMap(FeatureMap):
    """E4 -- EXTERNAL quantum self-attention control (QPSAN / QKSAN-style).

    A published-method baseline, NOT our contribution: a parameterised-quantum-
    circuit (PQC) attention *scoring* in the spirit of the Quantum (Probabilistic)
    Self-Attention Networks line of work --
        Zhao et al., "QKSAN: A Quantum Kernel Self-Attention Network",
            arXiv:2308.13422
        (and the QPSAN variant arXiv:2605.25365)
    -- where the query-key score is the overlap of two PQC-embedded states rather
    than a softmax dot-product. We re-cast that score as a *kernel feature map* so
    it drops onto the IDENTICAL phi(q)(phi(k)^T v) linear-attention path as every
    other map here, and so it is parameter-matchable (param-free, reuses the
    shared q/k/v Linears). The scientific question for E4 is whether this external
    quantum-style scoring ALSO merely ties classical RFF at matched budget.

    Faithful core (what we keep):
      * Quantum Logical Similarity (QKSAN): the attention score is the *fidelity*
        |<psi(q)|psi(k)>|^2 of two data-embedded states, not a raw dot product.
      * QPSAN PQC embedding: an RY angle embedding with `reuploads` (R) data
        re-uploading layers and a fixed CZ-ring entangler between layers --
        exactly the "PQC + data re-uploading" template both papers use.

    What we KEEP trainable (so this is a genuine *parameterised* external control,
    not a relabel of the fixed `quantum` map):
      * QPSAN's learnable variational rotation angles: a per-coordinate trainable
        `theta` (see __init__) that rescales each coordinate's encoding angle
        before the fixed sampled RY spectrum. The optimizer learns these angles,
        so the embedding ADAPTS during training (at init theta=1 reproduces the
        fixed angle map; afterwards it differs). Tiny param budget (dim_head per
        map, a handful per layer, faithful to QPSAN's "few params/layer").

    Documented simplifications (what we drop, for tractable + faithful-as-feasible):
      * The full QKSAN/QPSAN trainable *measurement observable* is fixed to the
        bare state fidelity (the standard "quantum kernel" reduction of a PQC
        attention head, Schuld 2021, arXiv:2101.11020: a PQC self-attention with a
        fixed measurement *is* a quantum-kernel attention). The trainable angles
        above are retained; only the measurement is fixed.
      * The CZ-ring entangler contributes a per-uploading-layer global phase that
        cancels in the fidelity |<.|.>|^2 for this product-RY embedding, so the
        squared overlap is separable and closed-form:

            K(x, x') = ( prod_j cos^2( bw*(x_j - x'_j) / 2 ) ) ** R
                     = ( prod_j (1 + cos(bw*(x_j - x'_j))) / 2 ) ** R .

        R re-uploads of an RY embedding therefore behave like R "layers" of the
        angle-embedding kernel -- the same separable family our `quantum` map
        uses, which is the POINT of E4: an external published quantum-attention
        scoring reduces (dequantizes) to a classical random-feature kernel, and
        we test it at matched budget on the same rig. The entangler is faithful to
        the circuit even though it leaves this particular fidelity separable; a
        reviewer can confirm the closed form against the statevector validator
        `qpsan_kernel_fidelity` / `qpsan_kernel_matrix`.

    Random features (so it rides the shared linear-attention path): since
        cos^2(t/2) = (1 + cos t)/2 = 1/2 + 1/4 e^{it} + 1/4 e^{-it},
    the per-coordinate spectrum (per re-upload) is the DISCRETE integer-harmonic
    lattice {0:1/2, +1:1/4, -1:1/4}; with R re-uploads each frequency is bw times
    a sum of R iid such draws. phi(x) = sqrt(2/r) cos(W x + b), W drawn from that
    lattice, b ~ U[0,2pi] -> <phi(x), phi(x')> = K(x, x') in expectation. (Same
    cos-feature machinery as ShiftInvariantRFF; kept as a standalone class so E4
    is unmistakably a SEPARATE external control with its own validator.)
    """

    def __init__(self, dim_head: int, num_features: int,
                 bandwidth: float | None = None, reuploads: int = 1,
                 dc: float = 0.0, qk_norm: bool = False):
        super().__init__(dim_head, num_features)
        self.qk_norm = qk_norm
        self.reuploads = reuploads
        self.dc = dc
        if bandwidth is None:
            bandwidth = 1.0 if qk_norm else dim_head ** -0.5
        self.bandwidth = bandwidth
        self.register_buffer('_w', self._sample_omega(), persistent=True)   # (r, d)
        self.register_buffer('_b', 2 * math.pi * torch.rand(num_features),  # (r,)
                             persistent=True)
        # TRAINABLE variational angles theta -- one learnable rotation per input
        # coordinate, the QPSAN/QKSAN "parameterised quantum self-attention"
        # learnable parameters. They rescale each coordinate's encoding angle
        # before the fixed sampled RY spectrum, so the embedding ADAPTS during
        # training. This is what makes E4 a genuine *parameterised* external
        # control rather than a relabel of the fixed `quantum` map: at step 0
        # theta=1 (the fixed angle map), and the optimizer learns per-coordinate
        # angles from there. Param budget = dim_head per map (a handful per
        # layer, faithful to QPSAN's "few params/layer"); the total is well under
        # 0.1% of the 1.8M budget and is reported as a ~matched budget.
        self.theta = nn.Parameter(torch.ones(dim_head))

    def _sample_omega(self) -> torch.Tensor:
        # each coord = bw * sum of R iid {0:1/2, +1:1/4, -1:1/4}  (RY-fidelity
        # spectrum, R re-uploads). Identical lattice to the angle-embedding
        # kernel -- that equivalence IS the E4 dequantization result.
        steps = torch.multinomial(
            torch.tensor([0.5, 0.25, 0.25]),
            self.num_features * self.dim_head * self.reuploads,
            replacement=True,
        ).view(self.num_features, self.dim_head, self.reuploads)
        vals = torch.tensor([0.0, 1.0, -1.0])[steps]                # {0,+1,-1}
        return self.bandwidth * vals.sum(dim=-1)                    # (r, d)

    def projection(self, device, dtype) -> torch.Tensor:
        return self._w.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        # x: (b, h, n, d) -> cos features (b, h, n, r [+1]). is_query unused
        # (the fidelity score is symmetric in q/k).
        if self.qk_norm:                                            # onto unit sphere
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        x = x * self.theta.to(x.dtype)                              # learnable angles
        w = self.projection(x.device, x.dtype)                      # (r, d)
        b = self._b.to(device=x.device, dtype=x.dtype)              # (r,)
        proj = torch.einsum('b h n d, r d -> b h n r', x, w) + b
        feats = math.sqrt(2.0 / self.num_features) * torch.cos(proj)
        if self.dc > 0:                                             # +ve denom floor
            pad = feats.new_full((*feats.shape[:-1], 1), math.sqrt(self.dc))
            feats = torch.cat([feats, pad], dim=-1)
        return feats


def qpsan_kernel_matrix(x: torch.Tensor, bandwidth: float = 1.0,
                        reuploads: int = 1) -> torch.Tensor:
    """Exact QPSAN/QKSAN fidelity score |<psi(x_i)|psi(x_j)>|^2 via statevector
    sim of the RY data-reuploading embedding (R re-uploads, CZ-ring entangler).
    Ground truth for `qpsan_kernel_fidelity`. Small d only (2^d amplitudes).

    The state per re-upload is the product RY embedding
        |psi(x)> = prod_j ( cos(bw*x_j/2)|0> + sin(bw*x_j/2)|1> )
    followed by a fixed CZ ring (which only flips amplitude signs, a fixed
    diagonal +-1 phase D). The fidelity is built explicitly from amplitudes so
    the validator does NOT assume the closed form it checks against the features.
    """
    n, d = x.shape
    idx = torch.arange(2 ** d)
    bits = ((idx.unsqueeze(1) >> torch.arange(d)) & 1).to(torch.float64)  # (2^d,d) {0,1}
    xc = x.to(torch.float64)
    # per-coordinate amplitudes: |0>->cos(bw x/2), |1>->sin(bw x/2)
    c = torch.cos(bandwidth * xc / 2)                                # (n, d)
    s = torch.sin(bandwidth * xc / 2)                                # (n, d)
    # amplitude of basis state `bits` = prod_j (bit? s_j : c_j)
    amp_c = torch.einsum('nd,bd->nbd', c, 1 - bits)                  # contribution where bit=0
    amp_s = torch.einsum('nd,bd->nbd', s, bits)                      # contribution where bit=1
    amp = (amp_c + amp_s).prod(dim=-1)                              # (n, 2^d) real amps
    # fixed CZ-ring: sign flip on basis states by parity of adjacent-pair ANDs.
    pair = (bits + bits.roll(-1, dims=-1)) >= 2                      # adjacent both-1
    sign = torch.where(pair.sum(dim=-1) % 2 == 0, 1.0, -1.0).to(torch.float64)  # (2^d,)
    psi = (amp * sign.unsqueeze(0))                                 # (n, 2^d), real
    gram = psi @ psi.t()                                           # <psi_i|psi_j> (real)
    fid = (gram.abs() ** 2)                                        # |<.|.>|^2, R=1
    return (fid ** reuploads).to(x.dtype)                          # R re-uploads


def qpsan_kernel_fidelity(x: torch.Tensor, num_features: int, bandwidth: float = 1.0,
                          reuploads: int = 1) -> dict:
    """Relative Frobenius error ||phi phi^T - K_qpsan|| / ||K_qpsan|| for one draw
    of r features -> 0 with r. Honest link between the E4 external-style features
    and the exact PQC fidelity score (the QPSAN analogue of Table 4)."""
    d = x.shape[-1]
    fm = QPSANFeatureMap(d, num_features, bandwidth=bandwidth, reuploads=reuploads)
    phi = fm(x.view(1, 1, *x.shape)).view(x.shape[0], num_features)
    approx = phi @ phi.t()
    exact = qpsan_kernel_matrix(x, bandwidth=bandwidth, reuploads=reuploads)
    err = torch.linalg.norm(approx - exact) / torch.linalg.norm(exact)
    return {"rel_fro_error": err.item(), "num_features": num_features,
            "reuploads": reuploads, "bandwidth": bandwidth}


def iqp_kernel_matrix(x: torch.Tensor, bandwidth: float = 1.0,
                      coupling: float = 1.0) -> torch.Tensor:
    """Exact ZZ/IQP quantum kernel via statevector sim (sum over all 2^d basis
    states). Ground truth for `iqp_kernel_fidelity`. Small d only (2^d states)."""
    n, d = x.shape
    idx = torch.arange(2 ** d)
    bits = (idx.unsqueeze(1) >> torch.arange(d)) & 1               # (2^d, d) {0,1}
    signs = (1 - 2 * bits).to(torch.float64)                      # {+1,-1}
    u = bandwidth * (x.to(torch.float64) @ signs.t())             # (n, 2^d): s.x
    phase = u + 0.5 * coupling * u * u                            # h(u)
    psi = torch.exp(1j * phase) / (2 ** (d / 2))                  # (n, 2^d) amplitudes
    gram = psi.conj() @ psi.t()                                   # <psi_i|psi_j>
    return (gram.abs() ** 2).to(x.dtype)                          # (n, n)


def iqp_kernel_fidelity(x: torch.Tensor, num_features: int, bandwidth: float = 1.0,
                        coupling: float = 1.0) -> dict:
    """Relative Frobenius error ||phi phi^T - K_iqp|| / ||K_iqp|| for one draw of
    r features -> 0 with r. Honest link between the classical features and the
    entangling quantum kernel (the entangled analogue of Table 4)."""
    d = x.shape[-1]
    fm = IQPKernelFeatureMap(d, num_features, bandwidth=bandwidth, coupling=coupling)
    phi = fm(x.view(1, 1, *x.shape)).view(x.shape[0], num_features)
    approx = phi @ phi.t()
    exact = iqp_kernel_matrix(x, bandwidth=bandwidth, coupling=coupling)
    err = torch.linalg.norm(approx - exact) / torch.linalg.norm(exact)
    return {"rel_fro_error": err.item(), "num_features": num_features,
            "coupling": coupling, "bandwidth": bandwidth}


def quantum_kernel_matrix(x: torch.Tensor, bandwidth: float = 1.0,
                          layers: int = 1) -> torch.Tensor:
    """Exact Gram matrix of the angle-embedding quantum kernel K(x_i, x_j).

    Closed form K = prod_d [(1 + cos(bandwidth*delta))/2] ** layers. This is the
    statevector-exact ground truth that the random features must converge to;
    `quantum_kernel_fidelity` checks ||phi phi^T - K|| -> 0 with r (Table 4).
    """
    delta = x.unsqueeze(-2) - x.unsqueeze(-3)                       # (n, n, d)
    per_coord = 0.5 * (1.0 + torch.cos(bandwidth * delta))         # (n, n, d)
    return (per_coord ** layers).prod(dim=-1)                       # (n, n)


def quantum_kernel_fidelity(x: torch.Tensor, num_features: int,
                            bandwidth: float = 1.0, layers: int = 1) -> dict:
    """Empirical link between the classical features and the quantum kernel.

    Returns the relative Frobenius error ||phi phi^T - K_quantum|| / ||K|| for a
    single random draw of r=`num_features` features. -> 0 as r grows (Table 4).
    """
    dim_head = x.shape[-1]
    fmap = QuantumKernelFeatureMap(dim_head, num_features, bandwidth=bandwidth,
                                   layers=layers)
    phi = fmap(x.view(1, 1, *x.shape)).view(x.shape[0], num_features)   # (n, r)
    approx = phi @ phi.t()                                              # (n, n)
    exact = quantum_kernel_matrix(x, bandwidth=bandwidth, layers=layers)
    err = torch.linalg.norm(approx - exact) / torch.linalg.norm(exact)
    return {"rel_fro_error": err.item(), "num_features": num_features,
            "layers": layers, "bandwidth": bandwidth}


# Backwards-compatible alias (the implemented method; was the stub name).
QuantumKernelRFF = QuantumKernelFeatureMap

# Keys follow the paper's variant names and the live builders (`hf_vit._make_feature_map`,
# `models.py`): "performer" is the FAVOR+ positive-feature map (softmax kernel);
# "gaussian_rff" is the paper's Gaussian-RFF baseline (RBF cos-RFF).
FEATURE_MAPS = {
    "performer": GaussianRFF,                # FAVOR+ positive features (softmax kernel)
    "gaussian_rff": GaussianRFFBochner,      # classical RBF cos-RFF (the paper's Gaussian RFF)
    "quantum": QuantumKernelFeatureMap,      # angle/product quantum-kernel map (depth L)
    "iqp": IQPKernelFeatureMap,              # entangling IQP quantum-kernel map (coupling c)
    "qpsan": QPSANFeatureMap,                # E4 external control (QKSAN/QPSAN-style); not reported in the paper
}
