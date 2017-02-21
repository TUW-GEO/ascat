.. _ascatformattable:

ASCAT BUFR format table
=======================

Taken from http://projects.knmi.nl/scatterometer/publications/pdf/ASCAT_Product_Manual.pdf

====== ========== ========================================================= ==============
Number Descriptor Parameter                                                 Unit
====== ========== ========================================================= ==============
1      001033     Identification Of Originating/Generating Centre           Code Table
2      001034     Identification Of Originating/Generating Sub-Centre       Code Table
3      025060     Software Identification                                   Numeric
4      001007     Satellite Identifier                                      Code Table
5      002019     Satellite Instruments                                     Code Table
6      001012     Direction Of Motion Of Moving Observing Platform          Degree True
7      004001     Year                                                      Year
8      004002     Month                                                     Month
9      004003     Day                                                       Day
10     004004     Hour                                                      Hour
11     004005     Minute                                                    Minute
12     004006     Second                                                    Second
13     005001     Latitude (High Accuracy)                                  Degree
14     006001     Longitude (High Accuracy)                                 Degree
15     005033     Pixel Size On Horizontal-1                                m
16     005040     Orbit Number                                              Numeric
17     006034     Cross Track Cell Number                                   Numeric
18     010095     Height Of Atmosphere Used                                 m
19     021157     Loss Per Unit Length Of Atmosphere Used                   dB/m
20     021150     Beam Collocation                                          Flag Table
====== ========== ========================================================= ==============

====== ========== ========================================================= ==============
Number Descriptor Parameter                                                 Unit
====== ========== ========================================================= ==============
21     008085     Beam Identifier                                           Code Table
22     002111     Radar Incidence Angle                                     Degree
23     002134     Antenna Beam Azimuth                                      Degree
24     021062     Backscatter                                               dB
25     021063     Radiometric Resolution (Noise Value)                      %
26     021158     ASCAT Kp Estimate Quality                                 Code Table
27     021159     ASCAT Sigma-0 Usability                                   Code Table
28     021160     ASCAT Use Of Synthetic Data                               Numeric
29     021161     ASCAT Synthetic Data Quality                              Numeric
30     021162     ASCAT Satellite Orbit And Attitude Quality                Numeric
31     021163     ASCAT Solar Array Reflection Contamination                Numeric
32     021164     ASCAT Telemetry Presence And Quality                      Numeric
33     021165     ASCAT Extrapolated Reference Function                     Numeric
34     021166     ASCAT Land Fraction                                       Numeric
====== ========== ========================================================= ==============

====== ========== ========================================================= ==============
Number Descriptor Parameter                                                 Unit
====== ========== ========================================================= ==============
35     008085     Beam Identifier                                           Code Table
36     002111     Radar Incidence Angle                                     Degree
37     002134     Antenna Beam Azimuth                                      Degree
38     021062     Backscatter                                               dB
39     021063     Radiometric Resolution (Noise Value)                      %
40     021158     ASCAT Kp Estimate Quality                                 Code Table
41     021159     ASCAT Sigma-0 Usability                                   Code Table
42     021160     ASCAT Use Of Synthetic Data                               Numeric
43     021161     ASCAT Synthetic Data Quality                              Numeric
44     021162     ASCAT Satellite Orbit And Attitude Quality                Numeric
45     021163     ASCAT Solar Array Reflection Contamination                Numeric
46     021164     ASCAT Telemetry Presence And Quality                      Numeric
47     021165     ASCAT Extrapolated Reference Function                     Numeric
48     021166     ASCAT Land Fraction                                       Numeric
====== ========== ========================================================= ==============

====== ========== ========================================================= ==============
Number Descriptor Parameter                                                 Unit
====== ========== ========================================================= ==============
49     008085     Beam Identifier                                           Code Table
50     002111     Radar Incidence Angle                                     Degree
51     002134     Antenna Beam Azimuth                                      Degree
52     021062     Backscatter                                               dB
53     021063     Radiometric Resolution (Noise Value)                      %
54     021158     ASCAT Kp Estimate Quality                                 Code Table
55     021159     ASCAT Sigma-0 Usability                                   Code Table
56     021160     ASCAT Use Of Synthetic Data                               Numeric
57     021161     ASCAT Synthetic Data Quality                              Numeric
58     021162     ASCAT Satellite Orbit And Attitude Quality                Numeric
59     021163     ASCAT Solar Array Reflection Contamination                Numeric
60     021164     ASCAT Telemetry Presence And Quality                      Numeric
61     021165     ASCAT Extrapolated Reference Function                     Numeric
62     021166     ASCAT Land Fraction                                       Numeric
====== ========== ========================================================= ==============

====== ========== ========================================================= ==============
Number Descriptor Parameter                                                 Unit
====== ========== ========================================================= ==============
63     025060     Software Identification                                   Numeric
64     025062     Database Identification                                   Numeric
65     040001     Surface Soil Moisture (Ms)                                %
66     040002     Estimated Error In Surface Soil Moisture                  %
67     021062     Backscatter                                               dB
68     021151     Estimated Error In Sigma0 At 40 Deg Incidence Angle       dB
69     021152     Slope At 40 Deg Incidence Angle                           dB/Degree
70     021153     Estimated Error In Slope At 40 Deg Incidence Angle        dB/Degree
71     021154     Soil Moisture Sensitivity                                 dB
72     021062     Dry Backscatter                                           dB
73     021088     Wet Backscatter                                           dB
74     040003     Mean Surface Soil Moisture                                Numeric
75     040004     Rain Fall Detection                                       Numeric
76     040005     Soil Moisture Correction Flag                             Flag Table
77     040006     Soil Moisture Processing Flag                             Flag Table
78     040007     Soil Moisture Quality                                     %
79     020065     Snow Cover                                                %
80     040008     Frozen Land Surface Fraction                              %
81     040009     Inundation And Wetland Fraction                           %
82     040010     Topographic Complexity                                    %
====== ========== ========================================================= ==============

====== ========== ========================================================= ==============
Number Descriptor Parameter                                                 Unit
====== ========== ========================================================= ==============
83     025060     Software Identification                                   Numeric
84     001032     Generating Application                                    Code Table
85     011082     Model Wind Speed At 10 m                                  m/s
86     011081     Model Wind Direction At 10 m                              Degree True
87     020095     Ice Probability                                           Numeric
88     020096     Ice Age (A-Parameter)                                     dB
89     021155     Wind Vector Cell Quality                                  Flag Table
90     021101     Number Of Vector Ambiguities                              Numeric
91     021102     Index Of Selected Wind Vector                             Numeric
92     031001     Delayed Descriptor Replication Factor                     Numeric
93     011012     Wind Speed At 10 m                                        m/s
94     011011     Wind Direction At 10 m                                    Degree True
95     021156     Backscatter Distance                                      Numeric
96     021104     Likelihood Computed For Solution                          Numeric
97     011012     Wind Speed At 10 m                                        m/s
98     011011     Wind Direction At 10 m                                    Degree True
99     021156     Backscatter Distance                                      Numeric
100    021104     Likelihood Computed For Solution                          Numeric
====== ========== ========================================================= ==============

Note that descriptor numbers 93-96 can be repeated 1 to 144 times, depending on the value
of the Delayed Descriptor Replication Factor (descriptor number 92)
