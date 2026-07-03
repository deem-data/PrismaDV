TABLE_PROFILING_EXAMPLE = """
Name:
  completeness: 1.0
  approximateNumDistinctValues: 48644
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram: null
Room Number:
  completeness: 1.0
  approximateNumDistinctValues: 386
  dataType: Integral
  typeCounts: {}
  isDataTypeInferred: false
  histogram: null
Hospital:
  completeness: 1.0
  approximateNumDistinctValues: 40036
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram: null
Insurance Provider:
  completeness: 1.0
  approximateNumDistinctValues: 5
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: Aetna
    count: 10913
    ratio: 0.197
  - value: Blue Cross
    count: 11059
    ratio: 0.199
  - value: UnitedHealthcare
    count: 11125
    ratio: 0.2
  - value: Cigna
    count: 11249
    ratio: 0.203
  - value: Medicare
    count: 11154
    ratio: 0.201
Billing Amount:
  completeness: 1.0
  approximateNumDistinctValues: 47735
  dataType: Fractional
  typeCounts: {}
  isDataTypeInferred: false
  histogram: null
Discharge Date:
  completeness: 1.0
  approximateNumDistinctValues: 1863
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram: null
Medication:
  completeness: 1.0
  approximateNumDistinctValues: 5
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: Aspirin
    count: 11094
    ratio: 0.2
  - value: Paracetamol
    count: 11071
    ratio: 0.199
  - value: Ibuprofen
    count: 11127
    ratio: 0.2
  - value: Penicillin
    count: 11068
    ratio: 0.199
  - value: Lipitor
    count: 11140
    ratio: 0.201
Admission Type:
  completeness: 1.0
  approximateNumDistinctValues: 3
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: Emergency
    count: 18269
    ratio: 0.329
  - value: Urgent
    count: 18576
    ratio: 0.335
  - value: Elective
    count: 18655
    ratio: 0.336
Medical Condition:
  completeness: 1.0
  approximateNumDistinctValues: 6
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: Hypertension
    count: 9245
    ratio: 0.167
  - value: Cancer
    count: 9227
    ratio: 0.166
  - value: Obesity
    count: 9231
    ratio: 0.166
  - value: Arthritis
    count: 9308
    ratio: 0.168
  - value: Diabetes
    count: 9304
    ratio: 0.168
  - value: Asthma
    count: 9185
    ratio: 0.165
Age:
  completeness: 1.0
  approximateNumDistinctValues: 79
  dataType: Integral
  typeCounts: {}
  isDataTypeInferred: false
  histogram: null
Date of Admission:
  completeness: 1.0
  approximateNumDistinctValues: 1824
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram: null
Test Results:
  completeness: 1.0
  approximateNumDistinctValues: 3
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: Abnormal
    count: 18627
    ratio: 0.336
  - value: Normal
    count: 18517
    ratio: 0.334
  - value: Inconclusive
    count: 18356
    ratio: 0.331
Blood Type:
  completeness: 1.0
  approximateNumDistinctValues: 8
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: A+
    count: 6956
    ratio: 0.125
  - value: O+
    count: 6917
    ratio: 0.125
  - value: B+
    count: 6945
    ratio: 0.125
  - value: AB+
    count: 6947
    ratio: 0.125
  - value: A-
    count: 6969
    ratio: 0.126
  - value: O-
    count: 6877
    ratio: 0.124
  - value: B-
    count: 6944
    ratio: 0.125
  - value: AB-
    count: 6945
    ratio: 0.125
Gender:
  completeness: 1.0
  approximateNumDistinctValues: 2
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram:
  - value: Male
    count: 27774
    ratio: 0.5
  - value: Female
    count: 27726
    ratio: 0.5
Doctor:
  completeness: 1.0
  approximateNumDistinctValues: 40792
  dataType: String
  typeCounts:
    Boolean: 0
    Fractional: 0
    Integral: 0
    Unknown: 0
    String: 55500
  isDataTypeInferred: false
  histogram: null
"""
