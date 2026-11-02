"""A horizontal tail separated out of the whole-aircraft derivatives, so its size can vary.

Classical tail contributions (Etkin & Reid ch. 2 and 5; Nelson ch. 2 and 4), with the tail
lift slope a_t, efficiency eta, downwash gradient eps_a, horizontal tail volume
V_H = S_t l_t / (S cbar), tail arm l_t from the CG, and elevator effectiveness tau:

    CL_a  = eta a_t (S_t/S) (1 - eps_a)      Cm_a  = -eta a_t V_H (1 - eps_a)
    CL_q  = 2 eta a_t V_H                      Cm_q  = -2 eta a_t V_H l_t/cbar
    CL_ad = 2 eta a_t V_H eps_a                Cm_ad = -2 eta a_t V_H (l_t/cbar) eps_a
    CL_de = eta tau a_t S_t/S                  Cm_de = -eta tau a_t V_H

Calibration from published derivatives needs one geometric assumption, the tail arm
l_t/cbar: the pitch damping is taken to be all tail, so eta a_t V_H = -Cm_q / (2 l_t/cbar);
then eps_a = Cm_adot / Cm_q and tau = -Cm_de / (eta a_t V_H). Scaling the tail area by k
adds (k - 1) times every tail contribution. The tail lift derivatives the source publishes
(the 747's CL_adot is negative, which no tail produces) are not used for the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..aircraft import Aircraft


@dataclass(frozen=True)
class HorizontalTail:
    eta_at_VH: float  # eta a_t V_H
    lt_c: float  # tail arm / mean chord (assumed)
    eps_a: float  # downwash gradient
    tau: float  # elevator effectiveness

    @classmethod
    def calibrate(cls, ac: Aircraft, lt_c: float) -> HorizontalTail:
        d = ac.aero
        k = -d.Cm_q / (2 * lt_c)
        return cls(eta_at_VH=k, lt_c=lt_c, eps_a=d.Cm_adot / d.Cm_q, tau=-d.Cm_de / k)

    def contributions(self) -> dict[str, float]:
        """Derivative contributions of this tail (whole-aircraft reference values)."""
        k, lt, e = self.eta_at_VH, self.lt_c, self.eps_a
        return dict(
            CL_a=k * (1 - e) / lt,
            Cm_a=-k * (1 - e),
            CL_q=2 * k,
            Cm_q=-2 * k * lt,
            CL_adot=2 * k * e,
            Cm_adot=-2 * k * lt * e,
            CL_de=self.tau * k / lt,
            Cm_de=-self.tau * k,
        )


def scale_tail(ac: Aircraft, tail: HorizontalTail, k: float) -> Aircraft:
    """The aircraft with its horizontal tail area multiplied by k (arm unchanged)."""
    d = ac.aero
    add = {name: getattr(d, name) + (k - 1.0) * v for name, v in tail.contributions().items()}
    return ac.with_(aero=d.with_(**add))
