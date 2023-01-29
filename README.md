# Wind-Turbine-Fault-Prediction

Repository of studies with the objective of Predicting Faults in Wind Turbines using machine learning.

## Data

Due to the existing limitations in the data access from a real wind turbine, the method developed used simulation software to generate the database for SVM training. The TurbSim software is a stochastic, full-field, turbulent-wind simulator responsible for generating wind time series to the FAST simulations . The FAST is a code responsible for wind turbine model dynamics, such as aerodynamics, structural, wind, electrical and control dynamics. The FAST wind turbine model was based on a GE 1.5s wind turbine and, already validated in research. The National Renewable Energy Laboratory (NREL) developed both software. FAST does not allow access to all electrical quantities of the generator, such as the currents required for SVM training and classification. To overcome this limitation, the authors used the MATLAB Simulink software to model a permanent magnet synchronous generator, making it possible to store the generator currents in a database. Table I shows the wind turbine characteristics. 

| Parameter            | Value                        |
|----------------------|------------------------------|
| Nominal Power        | 1.5 MW                       |
| Generator type       | permanent magnet synchronous |
| Hub height           | 84 m                         |
| Rotor diameter       | 70 m                         |
| Rotor orientation    | Upwind                       |
| Rotor configuration  | 3 blades                     |
| Nominal speed        | 2.14 rad/s (20 RPM)          |
| Nominal rotor torque | 736.79 kNm                   |

The database considered seven rotor imbalance conditions shown in Table II. This work generated a database containing 12 simulations for each wind speed and turbulence intensity combination presented in Table II. The wind time series are random, so the simulations are different using the same values of mean wind speed and turbulence intensity. The simulation time is 120 seconds but stored only the last 60 seconds, disregarding the initial transient of the three-phase currents of the generator (Ia, Ib and Ic) because according to, current signals are more reliable for monitoring the condition of WT. The developed method considered only the samples from the operating region 3 of the wind turbine because the power generation is nominal and the rotation speed is approximately constant.

| Parameter            | Value                                               |
|----------------------|-----------------------------------------------------|
| mean wind speed      | 15.0; 17.3; 19.5; 21.8; 24.0 (m/s)                  |
| Turbulence intensity | 5.0; 11.3; 17.5; 23.8; 30.0 (%)                     |
| Imbal. condition     | Balanced; mass (-3; +2 and +5%); aero (2; 3 and 4º) |
| Sampling frequency   | 2kHz                                                |
| Time simulation      | 120 seconds                                         |
| Operation region     | 3                                                   |
