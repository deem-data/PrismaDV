from .github_payments_handvalidation_base import GithubPaymentsHandValidationBase

# Based on code and data from https://github.com/sebastienrousseau/pain001/blob/main/pain001/csv/validate_csv_data.py
class GithubPaymentsHandValidationBatchBooking(GithubPaymentsHandValidationBase):


    def assumption_in_natural_language(self):
        return "Each value in the batch_booking column must be parseable as boolean string"

    def target_column(self):
        return "batch_booking"

    def ground_truth_constraint(self):
        return "hasPattern('batch_booking', '^\s*(?i:true|false)\s*$')"

    def data_to_pass(self):
        return  {
            "id": [1,2,3,4,5,6,7,8,9,10],
            "date": [
                "2025-08-20T09:00:00",
                "2025-08-20T09:15:00",
                "2025-08-20T09:30:00",
                "2025-08-20T09:45:00",
                "2025-08-20T10:00:00",
                "2025-08-20T10:15:00",
                "2025-08-20T10:30:00",
                "2025-08-20T10:45:00",
                "2025-08-20T11:00:00",
                "2025-08-20T11:15:00"
            ],
            "nb_of_txs": [1,2,1,3,1,2,1,4,1,2],
            "ctrl_sum": [1250.00, 1501.00, 499.99, 960.25, 320.75, 4990.00, 125.50, 860.75, 7420.00, 2999.95],
            "initiator_name": [
                "Alpha Corp", "Beta Industries", "Gamma Solutions", "Delta Traders", "Epsilon GmbH", "Zeta Holdings", "Eta Services", "Theta Technologies", "Iota Consulting", "Kappa Ventures"
            ],
            "payment_information_id": [
                "PMT20250820-001", "PMT20250820-002", "PMT20250820-003", "PMT20250820-004", "PMT20250820-005", "PMT20250820-006", "PMT20250820-007", "PMT20250820-008","PMT20250820-009", "PMT20250820-010"
            ],
            "payment_method": ["TRF","CHK","TRF","TRF","DD","TRF","TRF","CHK","TRF","DD"],
            "batch_booking": ["True", "false", "  True", "False", "True", "False", "True", "FalsE  ", "True", "True"],
            "service_level_code": ["SEPA","URGP","SEPA","SEPA","SEPA","URGP","SEPA","SEPA","SEPA","SEPA"],
            "requested_execution_date": [
                "2025-08-21","2025-08-21","2025-08-22","2025-08-22","2025-08-23",
                "2025-08-23","2025-08-24","2025-08-24","2025-08-25","2025-08-25"
            ],
            "debtor_name": [
                "Alpha Corp","Beta Industries","Gamma Solutions","Delta Traders","Epsilon GmbH",
                "Zeta Holdings","Eta Services","Theta Technologies","Iota Consulting","Kappa Ventures"
            ],
            "debtor_account_IBAN": [
                "DE89370400440532013000", "FR7630006000011234567890189", "NL91ABNA0417164300", "GB29NWBK60161331926819", "ES9121000418450200051332", "IT60X0542811101000000123456", "BE68539007547034", "PT50000201231234567890154", "DE12500105170648489890", "FR1420041010050500013M02606"
            ],
            "debtor_agent_BIC": [
                "COBADEFFXXX","BNPAFRPPXXX","INGBNL2AXXX","NWBKGB2L","CAIXESBBXXX", "BCITITMMXXX","KREDBEBB","CGDIPTPL","DEUTDEFFXXX","BNPAFRPPXXX"
            ],
            "forwarding_agent_BIC": [
                "DEUTDEFFXXX","SOGEFRPPXXX","ABNANL2AXXX","LOYDGB2L","BBVAESMMXXX", "UNCRITMM","AXABFRPP","BESZPTPL","DZBANKDEFFXXX","CMBRFR2BXXX"
            ],
            "charge_bearer": ["SLEV","DEBT","CRED","SHAR","SLEV","SHAR","DEBT","CRED","SLEV","SLEV"],
            "payment_id": [
                "E2E20250820-001", "E2E20250820-002", "E2E20250820-003", "E2E20250820-004", "E2E20250820-005", "E2E20250820-006", "E2E20250820-007", "E2E20250820-008", "E2E20250820-009", "E2E20250820-010"
            ],
            "payment_amount": [1250.00, 750.50, 499.99, 320.75, 999.99, 245.00, 15000.00, 85.50, 420.00, 2999.95],
            "currency": ["EUR","USD","EUR","GBP","EUR","EUR","CHF","EUR","USD","EUR"],
            "creditor_agent_BIC": [
                "BNPAFRPPXXX","INGBNL2AXXX","NWBKGB2L","DEUTDEFFXXX","BBVAESMMXXX",
                "BCITITMMXXX","KREDBEBB","CGDIPTPL","CMBRFR2BXXX","LOYDGB2L"
            ],
            "creditor_name": [
                "Omega Consulting","Acme Supplies BV","Tech Solutions Ltd","Berlin Printing GmbH","Hotel Riviera",
                "Quick Catering BV","Global Equipment Ltd","Taxi Service Lisbon","Paris Flowers SARL","IT Services UK"
            ],
            "creditor_account_IBAN": [
                "FR7630006000011234567890189", "NL91ABNA0417164300", "GB29NWBK60161331926819", "DE75512108001245126199", "ES7921000813610123456789","IT60X0542811101000000123457", "BE71096123456769", "PT50000201231234567890155", "FR7630004000031234567890147", "GB82WEST12345698765432"
            ],
            "remittance_information": [
                "Consulting services July", "Office supplies batch 12", "Software subscription", "Brochure print job", "Hotel conference booking", "Corporate catering", "Machinery purchase", "Transport reimbursement", "Corporate gifts", "Annual IT contract"
            ]
        }

    def data_to_reject(self):
        return  {
            "id": [1,2,3,4,5,6,7,8,9,10],
            "date": [
                "2025-08-20T09:00:00",
                "2025-08-20T09:15:00",
                "2025-08-20T09:30:00",
                "2025-08-20T09:45:00",
                "2025-08-20T10:00:00",
                "2025-08-20T10:15:00",
                "2025-08-20T10:30:00",
                "2025-08-20T10:45:00",
                "2025-08-20T11:00:00",
                "2025-08-20T11:15:00"
            ],
            "nb_of_txs": [1,2,1,3,1,2,1,4,1,2],
            "ctrl_sum": [1250.00, 1501.00, 499.99, 960.25, 320.75, 4990.00, 125.50, 860.75, 7420.00, 2999.95],
            "initiator_name": [
                "Alpha Corp", "Beta Industries", "Gamma Solutions", "Delta Traders", "Epsilon GmbH", "Zeta Holdings", "Eta Services", "Theta Technologies", "Iota Consulting", "Kappa Ventures"
            ],
            "payment_information_id": [
                "PMT20250820-001", "PMT20250820-002", "PMT20250820-003", "PMT20250820-004", "PMT20250820-005", "PMT20250820-006", "PMT20250820-007", "PMT20250820-008","PMT20250820-009", "PMT20250820-010"
            ],
            "payment_method": ["TRF","CHK","TRF","TRF","DD","TRF","TRF","CHK","TRF","DD"],
            "batch_booking": ["True", "False", "True", "False", "UNKNOWN", "False", "True", "False", "True", "True"],
            "service_level_code": ["SEPA","URGP","SEPA","SEPA","SEPA","URGP","SEPA","SEPA","SEPA","SEPA"],
            "requested_execution_date": [
                "2025-08-21","2025-08-21","2025-08-22","2025-08-22","2025-08-23",
                "2025-08-23","2025-08-24","2025-08-24","2025-08-25","2025-08-25"
            ],
            "debtor_name": [
                "Alpha Corp","Beta Industries","Gamma Solutions","Delta Traders","Epsilon GmbH",
                "Zeta Holdings","Eta Services","Theta Technologies","Iota Consulting","Kappa Ventures"
            ],
            "debtor_account_IBAN": [
                "DE89370400440532013000", "FR7630006000011234567890189", "NL91ABNA0417164300", "GB29NWBK60161331926819", "ES9121000418450200051332", "IT60X0542811101000000123456", "BE68539007547034", "PT50000201231234567890154", "DE12500105170648489890", "FR1420041010050500013M02606"
            ],
            "debtor_agent_BIC": [
                "COBADEFFXXX","BNPAFRPPXXX","INGBNL2AXXX","NWBKGB2L","CAIXESBBXXX", "BCITITMMXXX","KREDBEBB","CGDIPTPL","DEUTDEFFXXX","BNPAFRPPXXX"
            ],
            "forwarding_agent_BIC": [
                "DEUTDEFFXXX","SOGEFRPPXXX","ABNANL2AXXX","LOYDGB2L","BBVAESMMXXX", "UNCRITMM","AXABFRPP","BESZPTPL","DZBANKDEFFXXX","CMBRFR2BXXX"
            ],
            "charge_bearer": ["SLEV","DEBT","CRED","SHAR","SLEV","SHAR","DEBT","CRED","SLEV","SLEV"],
            "payment_id": [
                "E2E20250820-001", "E2E20250820-002", "E2E20250820-003", "E2E20250820-004", "E2E20250820-005", "E2E20250820-006", "E2E20250820-007", "E2E20250820-008", "E2E20250820-009", "E2E20250820-010"
            ],
            "payment_amount": [1250.00, 750.50, 499.99, 320.75, 999.99, 245.00, 15000.00, 85.50, 420.00, 2999.95],
            "currency": ["EUR","USD","EUR","GBP","EUR","EUR","CHF","EUR","USD","EUR"],
            "creditor_agent_BIC": [
                "BNPAFRPPXXX","INGBNL2AXXX","NWBKGB2L","DEUTDEFFXXX","BBVAESMMXXX",
                "BCITITMMXXX","KREDBEBB","CGDIPTPL","CMBRFR2BXXX","LOYDGB2L"
            ],
            "creditor_name": [
                "Omega Consulting","Acme Supplies BV","Tech Solutions Ltd","Berlin Printing GmbH","Hotel Riviera",
                "Quick Catering BV","Global Equipment Ltd","Taxi Service Lisbon","Paris Flowers SARL","IT Services UK"
            ],
            "creditor_account_IBAN": [
                "FR7630006000011234567890189", "NL91ABNA0417164300", "GB29NWBK60161331926819", "DE75512108001245126199", "ES7921000813610123456789","IT60X0542811101000000123457", "BE71096123456769", "PT50000201231234567890155", "FR7630004000031234567890147", "GB82WEST12345698765432"
            ],
            "remittance_information": [
                "Consulting services July", "Office supplies batch 12", "Software subscription", "Brochure print job", "Hotel conference booking", "Corporate catering", "Machinery purchase", "Transport reimbursement", "Corporate gifts", "Annual IT contract"
            ]
        }