* Initial State:
    * 65 km/s
    * Simulation starts 1 day from comet
    * Comet diameter: 0.69km
    * 1 m/s * 4 day initial separation of spacecraft, all converge on comet
    * Maneuvers at I-12h, 6h, 1h, 20min, 5min
* Errors:
    * Initial state: 100km, 2m/s per axis
    * Maneuver bias: Impulse bit of chosen thruster (2x 7.6e-4 m/s)
    * Maneuver scale: 2.5%
* Camera specs (DRACO):
    * Focal length: 2628mm
    * Aperture: 208mm
    * Sensor pitch: 6.5um
    * Sensor resolution: 2048x2048px
    * FOV: 0.29x0.29deg
    * Size: 311mm diameter, 590mm length
    * Mass: 9.55kg
    * Power: 4.95W
* Terminal guidance results (100k samples):
    * delta-v mean (3 sigma): 16 (39) m/s
    * impact diameter mean (3 sigma): 64 (410) m
* Other results:
    * Detectable at 1-1.5 days out with highest gain, 1s exposure time, worst case comet conditions (size, sun distance, phase angle)
    * With 800km flyby, 1.98m/px resolution, 1011px for 2km comet, 349px for 0.69km
