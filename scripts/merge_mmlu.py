import pandas as pd
import io

# Define the input data for all models (example data for demonstration)
data_cpt_k2_m12 = """
|                     Task                      |Metric|Value |   |Stderr|
|-----------------------------------------------|------|-----:|---|-----:|
|all                                            |qem   |0.2312|±  |0.0315|
|helm:mmlu:_average:5                           |qem   |0.2312|±  |0.0315|
|helm:mmlu:abstract_algebra:5                   |qem   |0.2200|±  |0.0416|
|helm:mmlu:anatomy:5                            |qem   |0.1852|±  |0.0336|
|helm:mmlu:astronomy:5                          |qem   |0.1776|±  |0.0311|
|helm:mmlu:business_ethics:5                    |qem   |0.3000|±  |0.0461|
|helm:mmlu:clinical_knowledge:5                 |qem   |0.2151|±  |0.0253|
|helm:mmlu:college_biology:5                    |qem   |0.2569|±  |0.0365|
|helm:mmlu:college_chemistry:5                  |qem   |0.2000|±  |0.0402|
|helm:mmlu:college_computer_science:5           |qem   |0.2600|±  |0.0441|
|helm:mmlu:college_mathematics:5                |qem   |0.2100|±  |0.0409|
|helm:mmlu:college_medicine:5                   |qem   |0.2081|±  |0.0310|
|helm:mmlu:college_physics:5                    |qem   |0.2157|±  |0.0409|
|helm:mmlu:computer_security:5                  |qem   |0.2800|±  |0.0451|
|helm:mmlu:conceptual_physics:5                 |qem   |0.2638|±  |0.0288|
|helm:mmlu:econometrics:5                       |qem   |0.2368|±  |0.0400|
|helm:mmlu:electrical_engineering:5             |qem   |0.2414|±  |0.0357|
|helm:mmlu:elementary_mathematics:5             |qem   |0.2090|±  |0.0209|
|helm:mmlu:formal_logic:5                       |qem   |0.2857|±  |0.0404|
|helm:mmlu:global_facts:5                       |qem   |0.1800|±  |0.0386|
|helm:mmlu:high_school_biology:5                |qem   |0.1774|±  |0.0217|
|helm:mmlu:high_school_chemistry:5              |qem   |0.1527|±  |0.0253|
|helm:mmlu:high_school_computer_science:5       |qem   |0.2500|±  |0.0435|
|helm:mmlu:high_school_european_history:5       |qem   |0.2182|±  |0.0323|
|helm:mmlu:high_school_geography:5              |qem   |0.1768|±  |0.0272|
|helm:mmlu:high_school_government_and_politics:5|qem   |0.1969|±  |0.0287|
|helm:mmlu:high_school_macroeconomics:5         |qem   |0.2026|±  |0.0204|
|helm:mmlu:high_school_mathematics:5            |qem   |0.2111|±  |0.0249|
|helm:mmlu:high_school_microeconomics:5         |qem   |0.2101|±  |0.0265|
|helm:mmlu:high_school_physics:5                |qem   |0.1987|±  |0.0326|
|helm:mmlu:high_school_psychology:5             |qem   |0.1927|±  |0.0169|
|helm:mmlu:high_school_statistics:5             |qem   |0.1528|±  |0.0245|
|helm:mmlu:high_school_us_history:5             |qem   |0.2500|±  |0.0304|
|helm:mmlu:high_school_world_history:5          |qem   |0.2700|±  |0.0289|
|helm:mmlu:human_aging:5                        |qem   |0.3139|±  |0.0311|
|helm:mmlu:human_sexuality:5                    |qem   |0.2595|±  |0.0384|
|helm:mmlu:international_law:5                  |qem   |0.2397|±  |0.0390|
|helm:mmlu:jurisprudence:5                      |qem   |0.2593|±  |0.0424|
|helm:mmlu:logical_fallacies:5                  |qem   |0.2209|±  |0.0326|
|helm:mmlu:machine_learning:5                   |qem   |0.3125|±  |0.0440|
|helm:mmlu:management:5                         |qem   |0.1748|±  |0.0376|
|helm:mmlu:marketing:5                          |qem   |0.2906|±  |0.0297|
|helm:mmlu:medical_genetics:5                   |qem   |0.3000|±  |0.0461|
|helm:mmlu:miscellaneous:5                      |qem   |0.2375|±  |0.0152|
|helm:mmlu:moral_disputes:5                     |qem   |0.2486|±  |0.0233|
|helm:mmlu:moral_scenarios:5                    |qem   |0.2380|±  |0.0142|
|helm:mmlu:nutrition:5                          |qem   |0.2255|±  |0.0239|
|helm:mmlu:philosophy:5                         |qem   |0.1865|±  |0.0221|
|helm:mmlu:prehistory:5                         |qem   |0.2160|±  |0.0229|
|helm:mmlu:professional_accounting:5            |qem   |0.2340|±  |0.0253|
|helm:mmlu:professional_law:5                   |qem   |0.2458|±  |0.0110|
|helm:mmlu:professional_medicine:5              |qem   |0.1838|±  |0.0235|
|helm:mmlu:professional_psychology:5            |qem   |0.2500|±  |0.0175|
|helm:mmlu:public_relations:5                   |qem   |0.2182|±  |0.0396|
|helm:mmlu:security_studies:5                   |qem   |0.1878|±  |0.0250|
|helm:mmlu:sociology:5                          |qem   |0.2438|±  |0.0304|
|helm:mmlu:us_foreign_policy:5                  |qem   |0.2800|±  |0.0451|
|helm:mmlu:virology:5                           |qem   |0.2831|±  |0.0351|
|helm:mmlu:world_religions:5                    |qem   |0.3216|±  |0.0358|
"""

data_cpt_k2_m6 = """
|                     Task                      |Metric|Value |   |Stderr|
|-----------------------------------------------|------|-----:|---|-----:|
|all                                            |qem   |0.2303|±  |0.0314|
|helm:mmlu:_average:5                           |qem   |0.2303|±  |0.0314|
|helm:mmlu:abstract_algebra:5                   |qem   |0.2200|±  |0.0416|
|helm:mmlu:anatomy:5                            |qem   |0.1852|±  |0.0336|
|helm:mmlu:astronomy:5                          |qem   |0.1776|±  |0.0311|
|helm:mmlu:business_ethics:5                    |qem   |0.3000|±  |0.0461|
|helm:mmlu:clinical_knowledge:5                 |qem   |0.2151|±  |0.0253|
|helm:mmlu:college_biology:5                    |qem   |0.2500|±  |0.0362|
|helm:mmlu:college_chemistry:5                  |qem   |0.2000|±  |0.0402|
|helm:mmlu:college_computer_science:5           |qem   |0.2600|±  |0.0441|
|helm:mmlu:college_mathematics:5                |qem   |0.1900|±  |0.0394|
|helm:mmlu:college_medicine:5                   |qem   |0.2081|±  |0.0310|
|helm:mmlu:college_physics:5                    |qem   |0.2157|±  |0.0409|
|helm:mmlu:computer_security:5                  |qem   |0.2800|±  |0.0451|
|helm:mmlu:conceptual_physics:5                 |qem   |0.2638|±  |0.0288|
|helm:mmlu:econometrics:5                       |qem   |0.2368|±  |0.0400|
|helm:mmlu:electrical_engineering:5             |qem   |0.2414|±  |0.0357|
|helm:mmlu:elementary_mathematics:5             |qem   |0.2090|±  |0.0209|
|helm:mmlu:formal_logic:5                       |qem   |0.2857|±  |0.0404|
|helm:mmlu:global_facts:5                       |qem   |0.1700|±  |0.0378|
|helm:mmlu:high_school_biology:5                |qem   |0.1774|±  |0.0217|
|helm:mmlu:high_school_chemistry:5              |qem   |0.1527|±  |0.0253|
|helm:mmlu:high_school_computer_science:5       |qem   |0.2500|±  |0.0435|
|helm:mmlu:high_school_european_history:5       |qem   |0.2121|±  |0.0319|
|helm:mmlu:high_school_geography:5              |qem   |0.1768|±  |0.0272|
|helm:mmlu:high_school_government_and_politics:5|qem   |0.1969|±  |0.0287|
|helm:mmlu:high_school_macroeconomics:5         |qem   |0.2026|±  |0.0204|
|helm:mmlu:high_school_mathematics:5            |qem   |0.2111|±  |0.0249|
|helm:mmlu:high_school_microeconomics:5         |qem   |0.2101|±  |0.0265|
|helm:mmlu:high_school_physics:5                |qem   |0.1987|±  |0.0326|
|helm:mmlu:high_school_psychology:5             |qem   |0.1908|±  |0.0168|
|helm:mmlu:high_school_statistics:5             |qem   |0.1528|±  |0.0245|
|helm:mmlu:high_school_us_history:5             |qem   |0.2500|±  |0.0304|
|helm:mmlu:high_school_world_history:5          |qem   |0.2700|±  |0.0289|
|helm:mmlu:human_aging:5                        |qem   |0.3139|±  |0.0311|
|helm:mmlu:human_sexuality:5                    |qem   |0.2595|±  |0.0384|
|helm:mmlu:international_law:5                  |qem   |0.2397|±  |0.0390|
|helm:mmlu:jurisprudence:5                      |qem   |0.2593|±  |0.0424|
|helm:mmlu:logical_fallacies:5                  |qem   |0.2209|±  |0.0326|
|helm:mmlu:machine_learning:5                   |qem   |0.3125|±  |0.0440|
|helm:mmlu:management:5                         |qem   |0.1748|±  |0.0376|
|helm:mmlu:marketing:5                          |qem   |0.2906|±  |0.0297|
|helm:mmlu:medical_genetics:5                   |qem   |0.3000|±  |0.0461|
|helm:mmlu:miscellaneous:5                      |qem   |0.2375|±  |0.0152|
|helm:mmlu:moral_disputes:5                     |qem   |0.2486|±  |0.0233|
|helm:mmlu:moral_scenarios:5                    |qem   |0.2380|±  |0.0142|
|helm:mmlu:nutrition:5                          |qem   |0.2255|±  |0.0239|
|helm:mmlu:philosophy:5                         |qem   |0.1865|±  |0.0221|
|helm:mmlu:prehistory:5                         |qem   |0.2130|±  |0.0228|
|helm:mmlu:professional_accounting:5            |qem   |0.2340|±  |0.0253|
|helm:mmlu:professional_law:5                   |qem   |0.2458|±  |0.0110|
|helm:mmlu:professional_medicine:5              |qem   |0.1838|±  |0.0235|
|helm:mmlu:professional_psychology:5            |qem   |0.2484|±  |0.0175|
|helm:mmlu:public_relations:5                   |qem   |0.2182|±  |0.0396|
|helm:mmlu:security_studies:5                   |qem   |0.1878|±  |0.0250|
|helm:mmlu:sociology:5                          |qem   |0.2438|±  |0.0304|
|helm:mmlu:us_foreign_policy:5                  |qem   |0.2800|±  |0.0451|
|helm:mmlu:virology:5                           |qem   |0.2831|±  |0.0351|
|helm:mmlu:world_religions:5                    |qem   |0.3216|±  |0.0358|
"""

data_sft_k2_m12 = """
|                     Task                      |Metric|Value |   |Stderr|
|-----------------------------------------------|------|-----:|---|-----:|
|all                                            |qem   |0.2239|±  |0.0311|
|helm:mmlu:_average:5                           |qem   |0.2239|±  |0.0311|
|helm:mmlu:abstract_algebra:5                   |qem   |0.2200|±  |0.0416|
|helm:mmlu:anatomy:5                            |qem   |0.1778|±  |0.0330|
|helm:mmlu:astronomy:5                          |qem   |0.1579|±  |0.0297|
|helm:mmlu:business_ethics:5                    |qem   |0.3000|±  |0.0461|
|helm:mmlu:clinical_knowledge:5                 |qem   |0.1925|±  |0.0243|
|helm:mmlu:college_biology:5                    |qem   |0.2569|±  |0.0365|
|helm:mmlu:college_chemistry:5                  |qem   |0.2000|±  |0.0402|
|helm:mmlu:college_computer_science:5           |qem   |0.2600|±  |0.0441|
|helm:mmlu:college_mathematics:5                |qem   |0.2000|±  |0.0402|
|helm:mmlu:college_medicine:5                   |qem   |0.2023|±  |0.0306|
|helm:mmlu:college_physics:5                    |qem   |0.2157|±  |0.0409|
|helm:mmlu:computer_security:5                  |qem   |0.2700|±  |0.0446|
|helm:mmlu:conceptual_physics:5                 |qem   |0.2596|±  |0.0287|
|helm:mmlu:econometrics:5                       |qem   |0.2368|±  |0.0400|
|helm:mmlu:electrical_engineering:5             |qem   |0.2345|±  |0.0353|
|helm:mmlu:elementary_mathematics:5             |qem   |0.2090|±  |0.0209|
|helm:mmlu:formal_logic:5                       |qem   |0.2698|±  |0.0397|
|helm:mmlu:global_facts:5                       |qem   |0.1800|±  |0.0386|
|helm:mmlu:high_school_biology:5                |qem   |0.1710|±  |0.0214|
|helm:mmlu:high_school_chemistry:5              |qem   |0.1379|±  |0.0243|
|helm:mmlu:high_school_computer_science:5       |qem   |0.2300|±  |0.0423|
|helm:mmlu:high_school_european_history:5       |qem   |0.2121|±  |0.0319|
|helm:mmlu:high_school_geography:5              |qem   |0.1717|±  |0.0269|
|helm:mmlu:high_school_government_and_politics:5|qem   |0.1813|±  |0.0278|
|helm:mmlu:high_school_macroeconomics:5         |qem   |0.1949|±  |0.0201|
|helm:mmlu:high_school_mathematics:5            |qem   |0.2111|±  |0.0249|
|helm:mmlu:high_school_microeconomics:5         |qem   |0.2059|±  |0.0263|
|helm:mmlu:high_school_physics:5                |qem   |0.1921|±  |0.0322|
|helm:mmlu:high_school_psychology:5             |qem   |0.1890|±  |0.0168|
|helm:mmlu:high_school_statistics:5             |qem   |0.1528|±  |0.0245|
|helm:mmlu:high_school_us_history:5             |qem   |0.2304|±  |0.0296|
|helm:mmlu:high_school_world_history:5          |qem   |0.2574|±  |0.0285|
|helm:mmlu:human_aging:5                        |qem   |0.3139|±  |0.0311|
|helm:mmlu:human_sexuality:5                    |qem   |0.2519|±  |0.0381|
|helm:mmlu:international_law:5                  |qem   |0.2314|±  |0.0385|
|helm:mmlu:jurisprudence:5                      |qem   |0.2407|±  |0.0413|
|helm:mmlu:logical_fallacies:5                  |qem   |0.2147|±  |0.0323|
|helm:mmlu:machine_learning:5                   |qem   |0.3036|±  |0.0436|
|helm:mmlu:management:5                         |qem   |0.1748|±  |0.0376|
|helm:mmlu:marketing:5                          |qem   |0.2906|±  |0.0297|
|helm:mmlu:medical_genetics:5                   |qem   |0.3000|±  |0.0461|
|helm:mmlu:miscellaneous:5                      |qem   |0.2363|±  |0.0152|
|helm:mmlu:moral_disputes:5                     |qem   |0.2312|±  |0.0227|
|helm:mmlu:moral_scenarios:5                    |qem   |0.2380|±  |0.0142|
|helm:mmlu:nutrition:5                          |qem   |0.2222|±  |0.0238|
|helm:mmlu:philosophy:5                         |qem   |0.1736|±  |0.0215|
|helm:mmlu:prehistory:5                         |qem   |0.2037|±  |0.0224|
|helm:mmlu:professional_accounting:5            |qem   |0.2305|±  |0.0251|
|helm:mmlu:professional_law:5                   |qem   |0.2216|±  |0.0106|
|helm:mmlu:professional_medicine:5              |qem   |0.1838|±  |0.0235|
|helm:mmlu:professional_psychology:5            |qem   |0.2418|±  |0.0173|
|helm:mmlu:public_relations:5                   |qem   |0.2182|±  |0.0396|
|helm:mmlu:security_studies:5                   |qem   |0.1755|±  |0.0244|
|helm:mmlu:sociology:5                          |qem   |0.2388|±  |0.0301|
|helm:mmlu:us_foreign_policy:5                  |qem   |0.2400|±  |0.0429|
|helm:mmlu:virology:5                           |qem   |0.2831|±  |0.0351|
|helm:mmlu:world_religions:5                    |qem   |0.3216|±  |0.0358|
"""

data_sft_k2_m6 = """
|                     Task                      |Metric|Value |   |Stderr|
|-----------------------------------------------|------|-----:|---|-----:|
|all                                            |qem   |0.2148|±  |0.0307|
|helm:mmlu:_average:5                           |qem   |0.2148|±  |0.0307|
|helm:mmlu:abstract_algebra:5                   |qem   |0.2100|±  |0.0409|
|helm:mmlu:anatomy:5                            |qem   |0.1778|±  |0.0330|
|helm:mmlu:astronomy:5                          |qem   |0.1711|±  |0.0306|
|helm:mmlu:business_ethics:5                    |qem   |0.2900|±  |0.0456|
|helm:mmlu:clinical_knowledge:5                 |qem   |0.2075|±  |0.0250|
|helm:mmlu:college_biology:5                    |qem   |0.2292|±  |0.0351|
|helm:mmlu:college_chemistry:5                  |qem   |0.1600|±  |0.0368|
|helm:mmlu:college_computer_science:5           |qem   |0.2500|±  |0.0435|
|helm:mmlu:college_mathematics:5                |qem   |0.1900|±  |0.0394|
|helm:mmlu:college_medicine:5                   |qem   |0.1850|±  |0.0296|
|helm:mmlu:college_physics:5                    |qem   |0.2157|±  |0.0409|
|helm:mmlu:computer_security:5                  |qem   |0.2600|±  |0.0441|
|helm:mmlu:conceptual_physics:5                 |qem   |0.2383|±  |0.0279|
|helm:mmlu:econometrics:5                       |qem   |0.2281|±  |0.0395|
|helm:mmlu:electrical_engineering:5             |qem   |0.2345|±  |0.0353|
|helm:mmlu:elementary_mathematics:5             |qem   |0.1878|±  |0.0201|
|helm:mmlu:formal_logic:5                       |qem   |0.2619|±  |0.0393|
|helm:mmlu:global_facts:5                       |qem   |0.1800|±  |0.0386|
|helm:mmlu:high_school_biology:5                |qem   |0.1645|±  |0.0211|
|helm:mmlu:high_school_chemistry:5              |qem   |0.1379|±  |0.0243|
|helm:mmlu:high_school_computer_science:5       |qem   |0.2400|±  |0.0429|
|helm:mmlu:high_school_european_history:5       |qem   |0.2061|±  |0.0316|
|helm:mmlu:high_school_geography:5              |qem   |0.1717|±  |0.0269|
|helm:mmlu:high_school_government_and_politics:5|qem   |0.1865|±  |0.0281|
|helm:mmlu:high_school_macroeconomics:5         |qem   |0.1872|±  |0.0198|
|helm:mmlu:high_school_mathematics:5            |qem   |0.2037|±  |0.0246|
|helm:mmlu:high_school_microeconomics:5         |qem   |0.2017|±  |0.0261|
|helm:mmlu:high_school_physics:5                |qem   |0.1854|±  |0.0317|
|helm:mmlu:high_school_psychology:5             |qem   |0.1633|±  |0.0158|
|helm:mmlu:high_school_statistics:5             |qem   |0.1343|±  |0.0233|
|helm:mmlu:high_school_us_history:5             |qem   |0.2304|±  |0.0296|
|helm:mmlu:high_school_world_history:5          |qem   |0.2447|±  |0.0280|
|helm:mmlu:human_aging:5                        |qem   |0.3139|±  |0.0311|
|helm:mmlu:human_sexuality:5                    |qem   |0.2443|±  |0.0377|
|helm:mmlu:international_law:5                  |qem   |0.2314|±  |0.0385|
|helm:mmlu:jurisprudence:5                      |qem   |0.2222|±  |0.0402|
|helm:mmlu:logical_fallacies:5                  |qem   |0.1902|±  |0.0308|
|helm:mmlu:machine_learning:5                   |qem   |0.2946|±  |0.0433|
|helm:mmlu:management:5                         |qem   |0.1553|±  |0.0359|
|helm:mmlu:marketing:5                          |qem   |0.2393|±  |0.0280|
|helm:mmlu:medical_genetics:5                   |qem   |0.2900|±  |0.0456|
|helm:mmlu:miscellaneous:5                      |qem   |0.2324|±  |0.0151|
|helm:mmlu:moral_disputes:5                     |qem   |0.2023|±  |0.0216|
|helm:mmlu:moral_scenarios:5                    |qem   |0.2380|±  |0.0142|
|helm:mmlu:nutrition:5                          |qem   |0.2255|±  |0.0239|
|helm:mmlu:philosophy:5                         |qem   |0.1415|±  |0.0198|
|helm:mmlu:prehistory:5                         |qem   |0.2037|±  |0.0224|
|helm:mmlu:professional_accounting:5            |qem   |0.1915|±  |0.0235|
|helm:mmlu:professional_law:5                   |qem   |0.2145|±  |0.0105|
|helm:mmlu:professional_medicine:5              |qem   |0.1838|±  |0.0235|
|helm:mmlu:professional_psychology:5            |qem   |0.2206|±  |0.0168|
|helm:mmlu:public_relations:5                   |qem   |0.1909|±  |0.0376|
|helm:mmlu:security_studies:5                   |qem   |0.1878|±  |0.0250|
|helm:mmlu:sociology:5                          |qem   |0.2438|±  |0.0304|
|helm:mmlu:us_foreign_policy:5                  |qem   |0.2800|±  |0.0451|
|helm:mmlu:virology:5                           |qem   |0.2651|±  |0.0344|
|helm:mmlu:world_religions:5                    |qem   |0.3041|±  |0.0353|
"""

data_baseline = """
|                     Task                      |Metric|Value |   |Stderr|
|-----------------------------------------------|------|-----:|---|-----:|
|all                                            |qem   |0.2931|±  |0.0339|
|helm:mmlu:_average:5                           |qem   |0.2931|±  |0.0339|
|helm:mmlu:abstract_algebra:5                   |qem   |0.3400|±  |0.0476|
|helm:mmlu:anatomy:5                            |qem   |0.2741|±  |0.0385|
|helm:mmlu:astronomy:5                          |qem   |0.3421|±  |0.0386|
|helm:mmlu:business_ethics:5                    |qem   |0.2300|±  |0.0423|
|helm:mmlu:clinical_knowledge:5                 |qem   |0.2566|±  |0.0269|
|helm:mmlu:college_biology:5                    |qem   |0.2500|±  |0.0362|
|helm:mmlu:college_chemistry:5                  |qem   |0.2700|±  |0.0446|
|helm:mmlu:college_computer_science:5           |qem   |0.3700|±  |0.0485|
|helm:mmlu:college_mathematics:5                |qem   |0.3200|±  |0.0469|
|helm:mmlu:college_medicine:5                   |qem   |0.2890|±  |0.0346|
|helm:mmlu:college_physics:5                    |qem   |0.3039|±  |0.0458|
|helm:mmlu:computer_security:5                  |qem   |0.3000|±  |0.0461|
|helm:mmlu:conceptual_physics:5                 |qem   |0.3319|±  |0.0308|
|helm:mmlu:econometrics:5                       |qem   |0.2895|±  |0.0427|
|helm:mmlu:electrical_engineering:5             |qem   |0.3862|±  |0.0406|
|helm:mmlu:elementary_mathematics:5             |qem   |0.3042|±  |0.0237|
|helm:mmlu:formal_logic:5                       |qem   |0.3016|±  |0.0410|
|helm:mmlu:global_facts:5                       |qem   |0.2600|±  |0.0441|
|helm:mmlu:high_school_biology:5                |qem   |0.2774|±  |0.0255|
|helm:mmlu:high_school_chemistry:5              |qem   |0.3202|±  |0.0328|
|helm:mmlu:high_school_computer_science:5       |qem   |0.2400|±  |0.0429|
|helm:mmlu:high_school_european_history:5       |qem   |0.3212|±  |0.0365|
|helm:mmlu:high_school_geography:5              |qem   |0.2525|±  |0.0310|
|helm:mmlu:high_school_government_and_politics:5|qem   |0.2487|±  |0.0312|
|helm:mmlu:high_school_macroeconomics:5         |qem   |0.2615|±  |0.0223|
|helm:mmlu:high_school_mathematics:5            |qem   |0.3556|±  |0.0292|
|helm:mmlu:high_school_microeconomics:5         |qem   |0.2563|±  |0.0284|
|helm:mmlu:high_school_physics:5                |qem   |0.2715|±  |0.0363|
|helm:mmlu:high_school_psychology:5             |qem   |0.2844|±  |0.0193|
|helm:mmlu:high_school_statistics:5             |qem   |0.3287|±  |0.0320|
|helm:mmlu:high_school_us_history:5             |qem   |0.3186|±  |0.0327|
|helm:mmlu:high_school_world_history:5          |qem   |0.2996|±  |0.0298|
|helm:mmlu:human_aging:5                        |qem   |0.2780|±  |0.0301|
|helm:mmlu:human_sexuality:5                    |qem   |0.4351|±  |0.0435|
|helm:mmlu:international_law:5                  |qem   |0.1901|±  |0.0358|
|helm:mmlu:jurisprudence:5                      |qem   |0.2778|±  |0.0433|
|helm:mmlu:logical_fallacies:5                  |qem   |0.3129|±  |0.0364|
|helm:mmlu:machine_learning:5                   |qem   |0.2500|±  |0.0411|
|helm:mmlu:management:5                         |qem   |0.3398|±  |0.0469|
|helm:mmlu:marketing:5                          |qem   |0.2607|±  |0.0288|
|helm:mmlu:medical_genetics:5                   |qem   |0.2100|±  |0.0409|
|helm:mmlu:miscellaneous:5                      |qem   |0.2669|±  |0.0158|
|helm:mmlu:moral_disputes:5                     |qem   |0.3150|±  |0.0250|
|helm:mmlu:moral_scenarios:5                    |qem   |0.2726|±  |0.0149|
|helm:mmlu:nutrition:5                          |qem   |0.3464|±  |0.0272|
|helm:mmlu:philosophy:5                         |qem   |0.3312|±  |0.0267|
|helm:mmlu:prehistory:5                         |qem   |0.2994|±  |0.0255|
|helm:mmlu:professional_accounting:5            |qem   |0.2411|±  |0.0255|
|helm:mmlu:professional_law:5                   |qem   |0.2301|±  |0.0108|
|helm:mmlu:professional_medicine:5              |qem   |0.2684|±  |0.0269|
|helm:mmlu:professional_psychology:5            |qem   |0.2533|±  |0.0176|
|helm:mmlu:public_relations:5                   |qem   |0.1545|±  |0.0346|
|helm:mmlu:security_studies:5                   |qem   |0.3673|±  |0.0309|
|helm:mmlu:sociology:5                          |qem   |0.3781|±  |0.0343|
|helm:mmlu:us_foreign_policy:5                  |qem   |0.3500|±  |0.0479|
|helm:mmlu:virology:5                           |qem   |0.3373|±  |0.0368|
|helm:mmlu:world_religions:5                    |qem   |0.2865|±  |0.0347|
"""

# Function to read Markdown data into a DataFrame
def read_md_table(data, value_col, stderr_col):
    df = pd.read_csv(io.StringIO(data), sep="|", skipinitialspace=True).drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"Value": value_col, "Stderr": stderr_col})
    return df

data_sft_k2_m6 = read_md_table(data_sft_k2_m6, "Value", "Stderr")
data_sft_k2_m12 = read_md_table(data_sft_k2_m12, "Value", "Stderr")
data_cpt_k2_m6 = read_md_table(data_cpt_k2_m6, "Value", "Stderr")
data_cpt_k2_m12 = read_md_table(data_cpt_k2_m12, "Value", "Stderr")
data_baseline = read_md_table(data_baseline, "Value", "Stderr")

# Trim extra spaces in the column names
data_sft_k2_m6.columns = data_sft_k2_m6.columns.str.strip()
data_sft_k2_m12.columns = data_sft_k2_m12.columns.str.strip()
data_cpt_k2_m6.columns = data_cpt_k2_m6.columns.str.strip()
data_cpt_k2_m12.columns = data_cpt_k2_m12.columns.str.strip()
data_baseline.columns = data_baseline.columns.str.strip()

data_sft_k2_m6.drop(columns=["Unnamed: 6"], inplace=True, errors="ignore")
data_sft_k2_m12.drop(columns=["Unnamed: 6"], inplace=True, errors="ignore")
data_cpt_k2_m6.drop(columns=["Unnamed: 6"], inplace=True, errors="ignore")
data_cpt_k2_m12.drop(columns=["Unnamed: 6"], inplace=True, errors="ignore")
data_baseline.drop(columns=["Unnamed: 6"], inplace=True, errors="ignore")

# Merge all models on common columns
merged_df = data_sft_k2_m12.merge(data_sft_k2_m6, on=["Task", "Metric"], how="outer", suffixes=(' SFT k=2, m=12', ' SFT k=2, m=6'))
merged_df = merged_df.merge(data_cpt_k2_m12, on=["Task", "Metric"], how="outer", suffixes=('', ' CPT k=2, m=12'))
merged_df = merged_df.merge(data_cpt_k2_m6, on=["Task", "Metric"], how="outer", suffixes=('', ' CPT k=2, m=6'))
merged_df = merged_df.merge(data_baseline, on=["Task", "Metric"], how="outer", suffixes=('', ' k=0, m=0 (Baseline)'))
if (merged_df.iloc[0].str.contains('-').any()):
    merged_df.drop(0, inplace=True)

# Check the merged data to verify
print(merged_df.head())

# Save the merged DataFrame to a new file (optional)
merged_df.to_csv('merged_models.csv', index=False)

# Display the final merged DataFrame
print(merged_df.head())
