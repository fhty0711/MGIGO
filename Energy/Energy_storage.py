"""
Energy Storage & Renewable Generation Models
=============================================

Pure physics-based models with NO artificial simplifications:
  - No binary charge/discharge variables → smooth σ-transitions instead
  - No linearization → real nonlinear efficiency curves
  - No KKT conditions → black-box optimizer handles non-convexity naturally

Piecewise functions are implemented via smooth σ-based blending
(reference: Functiontest/Hybridsystemtest.py §9).

Storage types:
  - Battery (Li-ion): C-rate dependent efficiency, cycle degradation
  - Hydrogen: electrolyzer + tank + fuel cell, nonlinear efficiency
  - Pumped Hydro: head-dependent turbine/pump efficiency
  - CAES: pressure-dependent compressor/expander

Renewable generation:
  - Solar PV: irradiance + temperature → power (piecewise, smoothed)
  - Wind Turbine: wind speed → power curve (cubic region, smoothed cut-in/cut-out)

Usage:
  from Energy.Energy_storage import Battery, SolarPV, WindTurbine

  batt = Battery(capacity=100.0, p_max=50.0, ...)
  p, soc = batt.step(x, prev_soc, dt)       # x ∈ ℝ → charge/discharge power
  solar = SolarPV(rated_power=10.0, ...)
  p = solar.power(irradiance, temperature)   # physics-based output
"""

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Optional, Tuple


# ======================================================================
# 1. Smooth transition utilities
# ======================================================================

def sigma_k(x: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Smooth saturation: σ_k(x) = kx / √(1 + (kx)²). Output ∈ (-1, 1)."""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx ** 2)


def smooth_step(x: jnp.ndarray, threshold: float = 0.0,
                beta: float = 10.0) -> jnp.ndarray:
    """Smooth step at threshold: ≈0 when x < threshold, ≈1 when x > threshold.

    σ(β·(x - threshold)) provides a continuous transition.
    β controls sharpness: β↑ → sharper step.
    """
    return 0.5 * (1.0 + sigma_k(x - threshold, k=beta))


def smooth_piecewise(x: jnp.ndarray,
                     breakpoints: jnp.ndarray,
                     values: jnp.ndarray,
                     beta: float = 10.0) -> jnp.ndarray:
    """Smooth piecewise function using σ-blending.

    Given breakpoints b_1 < b_2 < ... < b_{k-1} and values v_1, ..., v_k:
      f(x) smoothly transitions between v_i at each breakpoint.

    Implementation: weighted blend of adjacent segment values.
    """
    n_seg = len(values)
    result = jnp.zeros_like(x)
    for i in range(n_seg):
        # Weight for segment i: product of step functions that select this segment
        if i == 0:
            w = 1.0 - smooth_step(x, breakpoints[0], beta)
        elif i == n_seg - 1:
            w = smooth_step(x, breakpoints[-1], beta)
        else:
            w = (smooth_step(x, breakpoints[i-1], beta) *
                 (1.0 - smooth_step(x, breakpoints[i], beta)))
        result = result + w * values[i]
    return result


# ======================================================================
# 2. Battery Storage (Li-ion)
# ======================================================================

@dataclass(frozen=True)
class Battery:
    """Li-ion battery with nonlinear C-rate dependent efficiency.

    Encoding: single continuous variable x ∈ ℝ.
      P = P_max · tanh(x)           → charge/discharge power
      sign(P) determines direction (no binary variable needed).

    Efficiency model:
      η(P) = η_0 - η_loss · (|P| / P_max)²     (C-rate dependent loss)
      charge uses η_c, discharge uses η_d (typically asymmetric).

    Smooth mode blending:
      discharge_frac = smooth_step(P, 0, β)
      SoC change blends charge and discharge efficiency continuously.
    """
    capacity: float        # MWh (usable energy)
    p_max: float           # MW (max charge/discharge power)
    soc_min: float = 0.1   # min SoC (fraction, 0.1 = 10%)
    soc_max: float = 0.9   # max SoC (fraction, 0.9 = 90%)
    eta_c0: float = 0.95   # charge efficiency at low C-rate
    eta_d0: float = 0.95   # discharge efficiency at low C-rate
    eta_loss_c: float = 0.05  # efficiency loss at max C-rate (charge)
    eta_loss_d: float = 0.05  # efficiency loss at max C-rate (discharge)
    self_discharge: float = 0.0  # self-discharge rate per hour (fraction)
    cycle_cost: float = 0.0  # $/MWh degradation cost per cycle
    charge_cost_rate: float = 0.0  # $/MWh for charging operation (wear)
    discharge_cost_rate: float = 0.0  # $/MWh for discharging operation
    renewable_charge_discount: float = 0.5  # fraction — charging from RE is cheaper
    beta_mode: float = 5.0   # sharpness of charge/discharge blending

    @property
    def e_min(self) -> float:
        return self.soc_min * self.capacity

    @property
    def e_max(self) -> float:
        return self.soc_max * self.capacity

    def step(self, x: jnp.ndarray, prev_soc: jnp.ndarray,
             dt: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """One simulation step.

        Args:
            x: continuous control variable (ℝ → power setpoint)
            prev_soc: previous state of charge ∈ [0, 1]
            dt: time step (hours)

        Returns:
            power: actual power output (+ discharge, - charge) in MW
            new_soc: updated state of charge ∈ [0, 1]
        """
        # Power setpoint via tanh: maps ℝ → [-P_max, +P_max]
        p_setpoint = self.p_max * jnp.tanh(x)

        # SoC-dependent clipping
        max_discharge = (prev_soc - self.soc_min) * self.capacity / dt
        max_charge = (self.soc_max - prev_soc) * self.capacity / dt
        p = jnp.clip(p_setpoint, -max_charge, max_discharge)

        # Smooth charge/discharge blending (no binary switching!)
        # discharge_frac ≈ 1 when p > 0, ≈ 0 when p < 0
        discharge_frac = smooth_step(p, 0.0, self.beta_mode)

        # C-rate dependent efficiency
        c_rate = jnp.abs(p) / (self.p_max + 1e-6)
        eta_d = self.eta_d0 - self.eta_loss_d * c_rate ** 2
        eta_c = self.eta_c0 - self.eta_loss_c * c_rate ** 2

        # Smooth efficiency: blends between charge and discharge
        eta_eff = discharge_frac * eta_d + (1.0 - discharge_frac) * eta_c

        # Energy change:
        #   discharge (p>0): ΔE = -p * η_d * dt  (efficiency reduces output)
        #   charge   (p<0): ΔE = -p / η_c * dt  (efficiency increases input)
        energy_delta = -discharge_frac * p * dt * eta_d \
                       - (1.0 - discharge_frac) * p * dt / (eta_c + 1e-6)

        new_soc = prev_soc + energy_delta / self.capacity
        new_soc = new_soc - self.self_discharge * dt  # self-discharge
        new_soc = jnp.clip(new_soc, 0.0, 1.0)

        return p, new_soc

    def degradation_cost(self, p: jnp.ndarray, dt: float = 1.0) -> jnp.ndarray:
        """Cycle degradation cost: proportional to energy throughput."""
        if self.cycle_cost <= 0:
            return 0.0
        return self.cycle_cost * jnp.abs(p) * dt

    # ---- Cost decomposition ----

    def charge_cost(self, p_charge: jnp.ndarray,
                    from_renewable: bool = False,
                    dt: float = 1.0) -> jnp.ndarray:
        """Cost to CHARGE the battery at power p (p > 0 = MW absorbed).

        Components:
          1. Efficiency loss: (1/η_c - 1) * p → energy wasted as heat
          2. Operation wear: charge_cost_rate * p
          3. Cycle degradation: already in degradation_cost()
          4. Renewable discount: if from_renewable, reduce operation wear

        Returns total cost in $/h.
        """
        if self.charge_cost_rate <= 0 and self.eta_loss_c <= 0:
            return 0.0
        c_rate = jnp.abs(p_charge) / (self.p_max + 1e-6)
        eta_c = self.eta_c0 - self.eta_loss_c * c_rate ** 2
        efficiency_loss = jnp.abs(p_charge) * (1.0 / (eta_c + 1e-6) - 1.0)
        wear = self.charge_cost_rate * jnp.abs(p_charge)
        if from_renewable:
            wear = wear * (1.0 - self.renewable_charge_discount)
        return efficiency_loss + wear

    def discharge_cost(self, p_discharge: jnp.ndarray,
                       dt: float = 1.0) -> jnp.ndarray:
        """Cost to DISCHARGE the battery at power p (p > 0 = MW supplied).

        Components:
          1. Efficiency loss: (1 - η_d) * p → energy lost as heat
          2. Operation wear: discharge_cost_rate * p
          3. Cycle degradation: already in degradation_cost()
        """
        if self.discharge_cost_rate <= 0 and self.eta_loss_d <= 0:
            return 0.0
        c_rate = jnp.abs(p_discharge) / (self.p_max + 1e-6)
        eta_d = self.eta_d0 - self.eta_loss_d * c_rate ** 2
        efficiency_loss = jnp.abs(p_discharge) * (1.0 - eta_d)
        wear = self.discharge_cost_rate * jnp.abs(p_discharge)
        return efficiency_loss + wear

    def round_trip_cost(self, power: jnp.ndarray,
                        from_renewable: bool = False,
                        dt: float = 1.0) -> jnp.ndarray:
        """Total round-trip cost: charge + discharge + degradation."""
        return (self.charge_cost(power, from_renewable, dt) +
                self.discharge_cost(power, dt) +
                self.degradation_cost(power, dt))


# ======================================================================
# 3. Hydrogen Storage
# ======================================================================

@dataclass(frozen=True)
class HydrogenStorage:
    """Hydrogen energy storage: electrolyzer + tank + fuel cell.

    Encoding: single continuous variable x ∈ ℝ.
      x > 0 → electrolyzer produces H₂ (consumes power)
      x < 0 → fuel cell generates power (consumes H₂)
      Smooth blending via σ at x=0 (no binary!)

    Electrolyzer efficiency: η_e(P) nonlinear, best at 40-80% load.
    Fuel cell efficiency: η_f(P) nonlinear, best at 20-60% load.
    """
    electrolyzer_pmax: float   # MW (max input power)
    fuel_cell_pmax: float      # MW (max output power)
    tank_capacity: float       # MWh (H₂ energy equivalent)
    tank_soc_min: float = 0.05  # minimum tank level (fraction)
    tank_soc_max: float = 0.95  # maximum tank level (fraction)
    eta_e_peak: float = 0.70   # electrolyzer peak efficiency (LHV basis)
    eta_f_peak: float = 0.55   # fuel cell peak efficiency
    beta_mode: float = 5.0     # sharpness of electrolyzer/fuel-cell blending

    def step(self, x: jnp.ndarray, prev_tank_soc: jnp.ndarray,
             dt: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One simulation step.

        Returns:
            p_grid: power to/from grid (+ generation, - consumption) in MW
            h2_produced: hydrogen produced this step (MWh equivalent)
            new_tank_soc: updated tank state of charge ∈ [0, 1]
        """
        # Mode blending: electrolyzer_mode ≈ 1 when x > 0
        ely_mode = smooth_step(x, 0.0, self.beta_mode)

        # Electrolyzer power (x>0: consume power to make H₂)
        p_ely_raw = self.electrolyzer_pmax * jnp.tanh(x)
        p_ely = ely_mode * jnp.maximum(0.0, p_ely_raw)

        # Fuel cell power (x<0: consume H₂ to make power)
        p_fc_raw = self.fuel_cell_pmax * jnp.tanh(-x)
        p_fc = (1.0 - ely_mode) * jnp.maximum(0.0, p_fc_raw)

        # Electrolyzer efficiency: peaks at 60-80% load, drops at extremes
        ely_load = p_ely / (self.electrolyzer_pmax + 1e-6)
        eta_e = self.eta_e_peak * (1.0 - 0.3 * (ely_load - 0.7) ** 2)

        # Fuel cell efficiency: peaks at 30-50% load
        fc_load = p_fc / (self.fuel_cell_pmax + 1e-6)
        eta_f = self.eta_f_peak * (1.0 - 0.25 * (fc_load - 0.4) ** 2)

        # Energy flows
        h2_produced = p_ely * eta_e * dt           # H₂ energy added to tank
        h2_consumed = p_fc / (eta_f + 1e-6) * dt   # H₂ energy removed from tank

        # Tank limits
        max_fill = (self.tank_soc_max - prev_tank_soc) * self.tank_capacity
        max_draw = (prev_tank_soc - self.tank_soc_min) * self.tank_capacity

        h2_produced = jnp.minimum(h2_produced, max_fill)
        h2_consumed = jnp.minimum(h2_consumed, max_draw)

        new_tank_soc = prev_tank_soc + (h2_produced - h2_consumed) / self.tank_capacity
        new_tank_soc = jnp.clip(new_tank_soc, 0.0, 1.0)

        # Power to grid: fuel cell produces, electrolyzer consumes
        p_grid = p_fc - p_ely

        return p_grid, h2_produced, new_tank_soc


# ======================================================================
# 4. Pumped Hydro Storage
# ======================================================================

@dataclass(frozen=True)
class PumpedHydro:
    """Pumped hydro storage with head-dependent efficiency.

    Encoding: single continuous variable x ∈ ℝ.
      x > 0 → turbine mode (generate)
      x < 0 → pump mode (store)
      Smooth blending at x=0.
    """
    turbine_pmax: float      # MW (max generation)
    pump_pmax: float         # MW (max pumping power)
    upper_capacity: float    # MWh (upper reservoir energy equivalent)
    lower_capacity: float    # MWh (lower reservoir, typically much larger)
    head_base: float         # m (base hydraulic head)
    head_variation: float    # m (head change from full to empty)
    eta_turbine_peak: float = 0.90  # turbine peak efficiency
    eta_pump_peak: float = 0.88     # pump peak efficiency
    soc_min: float = 0.05    # min upper reservoir level
    soc_max: float = 0.95    # max upper reservoir level
    beta_mode: float = 5.0   # sharpness of turbine/pump blending

    def _head_factor(self, soc: jnp.ndarray) -> jnp.ndarray:
        """Head varies with upper reservoir level.

        Efficiency ∝ √(head) for turbine, modified Francis turbine model.
        """
        head = self.head_base + self.head_variation * (soc - 0.5)
        head_ratio = head / self.head_base
        # Turbine efficiency degrades ~2% per 10% head change
        return 1.0 - 0.2 * (1.0 - head_ratio) ** 2

    def step(self, x: jnp.ndarray, prev_soc: jnp.ndarray,
             dt: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """One simulation step.

        Returns:
            p_grid: power to grid (+ generate, - pump) in MW
            new_soc: updated upper reservoir state of charge ∈ [0, 1]
        """
        turbine_mode = smooth_step(x, 0.0, self.beta_mode)

        # Turbine generation (x>0)
        p_turbine = turbine_mode * self.turbine_pmax * jnp.tanh(x)
        p_turbine = jnp.maximum(0.0, p_turbine)

        # Pump consumption (x<0)
        p_pump = (1.0 - turbine_mode) * self.pump_pmax * jnp.tanh(-x)
        p_pump = jnp.maximum(0.0, p_pump)

        # Head-dependent efficiency
        hf = self._head_factor(prev_soc)
        eta_t = self.eta_turbine_peak * hf
        eta_p = self.eta_pump_peak * hf

        # Energy change in upper reservoir
        e_discharge = p_turbine / (eta_t + 1e-6) * dt  # water released
        e_charge = p_pump * eta_p * dt                   # water pumped up

        max_draw = (prev_soc - self.soc_min) * self.upper_capacity
        max_fill = (self.soc_max - prev_soc) * self.upper_capacity

        e_discharge = jnp.minimum(e_discharge, max_draw)
        e_charge = jnp.minimum(e_charge, max_fill)

        new_soc = prev_soc + (e_charge - e_discharge) / self.upper_capacity
        new_soc = jnp.clip(new_soc, 0.0, 1.0)

        p_grid = p_turbine - p_pump
        return p_grid, new_soc


# ======================================================================
# 5. Compressed Air Energy Storage (CAES)
# ======================================================================

@dataclass(frozen=True)
class CAES:
    """Compressed Air Energy Storage (diabatic or adiabatic).

    Encoding: single continuous variable x ∈ ℝ.
      x > 0 → expander mode (generate)
      x < 0 → compressor mode (store)
    """
    expander_pmax: float     # MW (max generation)
    compressor_pmax: float   # MW (max compression power)
    storage_capacity: float  # MWh (energy equivalent)
    pressure_max: float      # bar (max cavern pressure)
    pressure_min: float      # bar (min cavern pressure)
    eta_expander: float = 0.80   # expander efficiency
    eta_compressor: float = 0.82  # compressor efficiency
    heat_rate: float = 4.0       # MWh_heat / MWh_electric for diabatic
    soc_min: float = 0.05
    soc_max: float = 0.95
    beta_mode: float = 5.0

    def _pressure_factor(self, soc: jnp.ndarray) -> jnp.ndarray:
        """Pressure ratio affects compressor work.

        Higher cavern pressure → more compression work needed.
        Model: work ∝ ln(P_cavern / P_ambient) / ln(P_max / P_ambient).
        """
        P_ambient = 1.0  # bar
        pressure = self.pressure_min + (self.pressure_max - self.pressure_min) * soc
        # Normalized compression ratio effect
        ratio = (jnp.log(pressure / P_ambient) /
                 jnp.log(self.pressure_max / P_ambient) + 1e-6)
        return jnp.clip(ratio, 0.5, 1.2)

    def step(self, x: jnp.ndarray, prev_soc: jnp.ndarray,
             dt: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One simulation step.

        Returns:
            p_grid: power to grid (+ generate, - compress) in MW
            heat_required: heat needed for expansion (diabatic, MW thermal)
            new_soc: updated storage state of charge ∈ [0, 1]
        """
        expander_mode = smooth_step(x, 0.0, self.beta_mode)

        p_expand = expander_mode * self.expander_pmax * jnp.tanh(x)
        p_expand = jnp.maximum(0.0, p_expand)

        p_compress = (1.0 - expander_mode) * self.compressor_pmax * jnp.tanh(-x)
        p_compress = jnp.maximum(0.0, p_compress)

        pf = self._pressure_factor(prev_soc)

        # Energy flows
        e_generated = p_expand * dt / (self.eta_expander * pf + 1e-6)  # air consumed
        e_stored = p_compress * self.eta_compressor * pf * dt          # air compressed

        max_draw = (prev_soc - self.soc_min) * self.storage_capacity
        max_fill = (self.soc_max - prev_soc) * self.storage_capacity

        e_generated = jnp.minimum(e_generated, max_draw)
        e_stored = jnp.minimum(e_stored, max_fill)

        new_soc = prev_soc + (e_stored - e_generated) / self.storage_capacity
        new_soc = jnp.clip(new_soc, 0.0, 1.0)

        p_grid = p_expand - p_compress
        heat_required = p_expand * self.heat_rate  # for diabatic CAES

        return p_grid, heat_required, new_soc


# ======================================================================
# 6. Solar PV
# ======================================================================

@dataclass(frozen=True)
class SolarPV:
    """Photovoltaic panel with physics-based single-diode model.

    Key: no artificial simplifications. The power curve uses smooth
    σ-transitions instead of hard if-else for cut-in/cut-out thresholds.

    Power output depends on:
      - Irradiance G (W/m²)
      - Cell temperature T_cell (°C)
      - Incidence angle (simplified: cos(θ) factor)
    """
    rated_power: float       # kWp (peak power at STC: 1000 W/m², 25°C)
    area: float = 0.0        # m² (panel area, optional — computed from efficiency if 0)
    efficiency_stc: float = 0.20  # module efficiency at STC
    temp_coeff: float = -0.0035  # 1/°C (power temperature coefficient, typical -0.35%/°C)
    noct: float = 45.0       # °C (Nominal Operating Cell Temperature)
    beta_low_light: float = 10.0  # sharpness of low-light transition
    # Economics
    o_and_m: float = 0.0     # $/MWh O&M cost (marginal, essentially 0 for solar)
    curtailment_cost: float = 0.0  # $/MWh penalty for spilling (0 = free curtailment)

    def __post_init__(self):
        if self.area <= 0:
            # Compute from rated power and STC efficiency
            object.__setattr__(self, 'area',
                               self.rated_power / (1.0 * self.efficiency_stc))

    def power(self, irradiance: jnp.ndarray,
              temp_ambient: jnp.ndarray = 25.0,
              cos_theta: jnp.ndarray = 1.0) -> jnp.ndarray:
        """Compute PV power output.

        Args:
            irradiance: global irradiance on panel plane (W/m²)
            temp_ambient: ambient temperature (°C)
            cos_theta: cosine of incidence angle

        Returns:
            power: electrical output (kW)

        Physics:
          T_cell = T_ambient + (NOCT - 20) * G / 800
          P = P_rated * (G / 1000) * [1 + γ·(T_cell - 25)] * cos_theta

        With smooth low-light cutoff (no hard if-else):
          The ratio G/1000 already handles low irradiance naturally.
          No binary cutoff needed — power → 0 smoothly as G → 0.
        """
        # Cell temperature (NOCT model)
        t_cell = temp_ambient + (self.noct - 20.0) * irradiance / 800.0

        # Temperature-corrected efficiency
        temp_factor = 1.0 + self.temp_coeff * (t_cell - 25.0)

        # Power: linear with irradiance, no artificial cut-in/cut-out
        p_stc = self.rated_power * irradiance / 1000.0

        # Smooth low-light behavior (very low irradiance → negligible power)
        # No hard cutoff — σ transition ensures continuous behavior
        low_light_factor = smooth_step(irradiance, 5.0, self.beta_low_light)

        power = p_stc * temp_factor * cos_theta * low_light_factor
        return jnp.maximum(0.0, power)

    # ---- Economic interface ----

    def direct_cost(self, power_mw: float) -> float:
        """Cost to feed solar power directly to grid ($/h).

        Only O&M — essentially zero marginal cost.
        """
        return self.o_and_m * power_mw

    def to_storage_loss(self, power_mw: float) -> float:
        """Energy loss when routing solar → storage → grid.

        Returns fractional loss (0.05 = 5% conversion penalty).
        Storage always has round-trip losses; routing through it
        is slightly worse than direct grid feed due to extra conversion.
        """
        return 0.02 * power_mw  # 2% converter loss for DC-coupled PV+battery

    def curtail_cost(self, power_mw: float) -> float:
        return self.curtailment_cost * power_mw


# ======================================================================
# 7. Wind Turbine
# ======================================================================

@dataclass(frozen=True)
class WindTurbine:
    """Wind turbine with smooth power curve (no piecewise if-else!).

    Standard power curve regions (traditionally piecewise):
      1. v < v_cutin:        P = 0
      2. v_cutin ≤ v < v_rated: P = P_rated · (v³ - v_cutin³)/(v_rated³ - v_cutin³)
      3. v_rated ≤ v < v_cutout: P = P_rated
      4. v ≥ v_cutout:       P = 0

    Smooth implementation: blend regions with σ transitions.
    No binary cut-in/cut-out — solver sees continuous function.
    """
    rated_power: float       # kW
    v_cutin: float = 3.0     # m/s (cut-in wind speed)
    v_rated: float = 12.0    # m/s (rated wind speed)
    v_cutout: float = 25.0   # m/s (cut-out wind speed)
    hub_height: float = 80.0 # m
    beta_transition: float = 2.0  # sharpness of σ-transitions at cut-in/cut-out
    # Economics
    o_and_m: float = 2.0     # $/MWh O&M (wind has minor mechanical wear)
    curtailment_cost: float = 0.0  # $/MWh spilling cost

    def power(self, wind_speed: jnp.ndarray,
              air_density: float = 1.225) -> jnp.ndarray:
        """Compute wind turbine power via smooth piecewise blending.

        Args:
            wind_speed: wind speed at hub height (m/s)
            air_density: air density (kg/m³), default 1.225 at sea level

        Returns:
            power: electrical output (kW)
        """
        v = wind_speed

        # Region weights via smooth σ-transitions (no if-else!)
        entering = smooth_step(v, self.v_cutin, self.beta_transition)
        in_rated = smooth_step(v, self.v_rated, self.beta_transition)
        exiting  = smooth_step(self.v_cutout, v, self.beta_transition)  # ≈1 when v < cutout

        # Cubic region power (region 2)
        v_cube = jnp.clip(v, 0.0, self.v_rated)  # cubically interpolate up to rated
        p_cubic = self.rated_power * (
            (v_cube ** 3 - self.v_cutin ** 3) /
            (self.v_rated ** 3 - self.v_cutin ** 3 + 1e-6)
        )

        # Smooth blending of all regions:
        #   region 1 (below cutin): weight = (1-entering)
        #   region 2 (cubic):       weight = entering * (1-in_rated) * exiting
        #   region 3 (rated):       weight = entering * in_rated * exiting
        #   region 4 (above cutout): weight = (1-exiting)
        w_cubic = entering * (1.0 - in_rated) * exiting
        w_rated = entering * in_rated * exiting

        power = w_cubic * p_cubic + w_rated * self.rated_power

        # Air density correction (rated at 1.225 kg/m³)
        power = power * air_density / 1.225

        return jnp.maximum(0.0, power)

    # ---- Economic interface ----

    def direct_cost(self, power_mw: float) -> float:
        """Cost to feed wind power directly to grid ($/h)."""
        return self.o_and_m * power_mw

    def to_storage_loss(self, power_mw: float) -> float:
        """Energy loss routing wind → storage (AC/DC conversion + storage wear).

        Slightly higher than solar due to AC-DC-AC round-trip.
        """
        return 0.04 * power_mw  # 4% conversion penalty for wind+storage

    def curtail_cost(self, power_mw: float) -> float:
        return self.curtailment_cost * power_mw


# ======================================================================
# 8. Combined Renewable + Storage System
# ======================================================================

@dataclass(frozen=True)
class EnergyHub:
    """Simple energy hub combining generation + storage + load.

    Not a full model — just a convenience container for composing
    solar, wind, storage, and generators into a single cost function.
    """
    generators: 'GeneratorSet' = None   # from Generator.py
    battery: Battery = None
    solar: SolarPV = None
    wind: WindTurbine = None
    hydrogen: HydrogenStorage = None
    pumped_hydro: PumpedHydro = None

    def total_generation(self, x_gen, prev_gen_power,
                         irradiance, wind_speed, temp,
                         x_storage, prev_soc, demand, dt):
        """Compute total net generation from all assets.

        This is a template — actual usage wraps this in Constran build().
        """
        total = 0.0

        # Thermal generators
        if self.generators is not None:
            gen_p = self.generators.decode_incremental(x_gen, prev_gen_power)
            gen_on = self.generators.on_mask(gen_p)
            total = total + jnp.sum(gen_p * gen_on)

        # Solar
        if self.solar is not None:
            total = total + self.solar.power(irradiance, temp) / 1000.0  # kW→MW

        # Wind
        if self.wind is not None:
            total = total + self.wind.power(wind_speed) / 1000.0

        # Battery
        if self.battery is not None:
            p_batt, new_soc = self.battery.step(x_storage, prev_soc, dt)
            total = total + p_batt  # + discharge, - charge

        return total


# ======================================================================
# 9. Self-test (run directly to validate model physics)
# ======================================================================

if __name__ == "__main__":
    import numpy as np
    print("Models defined — see Energy/test_models.py for validation tests.")
    print("Run:  uv run python Energy/test_models.py")
