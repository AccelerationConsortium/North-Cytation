## powder_settings.py

from dataclasses import dataclass


@dataclass
class PowderSettings:
    opening_deg: int  # (0, 90] how many degrees to open the valve during dispense
    percent_target: float  # (0, 1] dispense iter target is percent_target*remaining
    max_growth: float  # how much the dispense time can grow/shrink between iters (i.e. 2x)
    thresh : float = 10  # settings take effect below this amount remaining - unless there is a lower one
    amplitude: int = 100  # (0, 100] vibratory amplitude in %
    freq: int = 80  # vibratory freq in hz
    max_shake_t: int = 1000 # max shake time in ms
    min_shake_t: int = 25 # min shake time in ms
    shut_valve: bool = True  # shut valve between every iteration - needed for free-flowing powders

    
@dataclass
class PowderProtocol:
    tol : float  # best expected dispense tolerance in units of mg
    fast_settings: PowderSettings
    med_settings: PowderSettings
    slow_settings: PowderSettings
    scale_delay: float = 0.5  # seconds before measuring powder... fall time for small amounts of solids
    

default_ps = PowderProtocol(tol = 0.2,
                        fast_settings = PowderSettings(
                            opening_deg = 50,
                            percent_target = 0.75,
                            max_growth = 3
                            ),
                        med_settings = PowderSettings(
                            thresh = 5,
                            opening_deg = 30,
                            percent_target = 0.50,
                            max_growth = 1.25,
                            amplitude = 75
                            ),
                        slow_settings = PowderSettings(
                            thresh = 3,
                            opening_deg = 25,
                            percent_target = 1,
                            max_growth = 1.1,
                            shut_valve = False
                            ),
                         scale_delay=1
                        )
                            
sticky_ps = PowderProtocol(tol = 0.2,
                        fast_settings = PowderSettings(
                            opening_deg = 40,
                            percent_target = 0.5,
                            max_growth = 2,
                            freq = 80,
                            shut_valve = False
                            ),
                        med_settings = PowderSettings(
                            thresh = 10,
                            opening_deg = 35,
                            percent_target = 0.50,
                            max_growth = 1.25,
                            freq = 80,
                            amplitude = 90,
                            shut_valve = False,
                            ),
                        slow_settings = PowderSettings(
                            thresh = 3,
                            opening_deg = 25,
                            percent_target = 0.5,
                            max_growth = 1.1,
                            shut_valve = False,
                            freq = 80,
                            amplitude = 70,
                            min_shake_t = 100,
                            max_shake_t = 250
                            ),
                         scale_delay=2
                        )

friendly_ps = PowderProtocol(tol = 0.2,
                        fast_settings = PowderSettings(
                            opening_deg = 45,
                            percent_target = 0.5,
                            max_growth = 2,
                            shut_valve = False
                            ),
                        med_settings = PowderSettings(
                            thresh = 10,
                            opening_deg = 30,
                            percent_target = 0.50,
                            max_growth = 1.25,
                            shut_valve = False,
                            amplitude=80
                            ),
                        slow_settings = PowderSettings(
                            thresh = 2,
                            opening_deg = 25,
                            percent_target = 0.25,
                            max_growth = 1.1,
                            shut_valve = False,
                            amplitude = 70
                            ),
                         scale_delay=2.5
                        )

flowing_ps = PowderProtocol(tol = 0.2,
                        fast_settings = PowderSettings(
                            opening_deg = 45,
                            percent_target = 0.5,
                            max_growth = 2,
                            shut_valve = True
                            ),
                        med_settings = PowderSettings(
                            thresh = 10,
                            opening_deg = 40,
                            percent_target = 0.50,
                            max_growth = 1.25,
                            shut_valve = True
                            ),
                        slow_settings = PowderSettings(
                            thresh = 2,
                            opening_deg = 30,
                            percent_target = 0.25,
                            max_growth = 1.1,
                            shut_valve = True
                            ),
                         scale_delay=2.5
                        )
