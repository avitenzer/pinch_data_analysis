1) Currency Normalization: Converted all formats to consistent floats

"$-1,027.95" -> -1027.95
"4.385,12" -> 4385.12 (European format)
"€3,631.24" -> 3631.24

We also added the ccy transaction column

2) Date Standardization: Unified all date formats to YYYY-MM-DD

"12-Oct-2024 09:43:59 UTC" -> 2024-10-12
"25/09/2023" -> 2023-09-24
"September 22, 2023" -> 2023-09-21


3) Column Standardization: Clean, consistent naming

"User ID" → user_id
"Settlement Date" → settlement_date

4) Quality Flagging: Identifies problematic rows and flagging them

Missing critical fields
Invalid amounts
No valid dates

5) when we have variation of merchant names we set it to the correct name i.e. in the messy data Amazon can be amazon, ama on, ama#on or amaxon

6) If we dont have the merchant name we try to see if the merchant name is available in the transaction description and we set it accordingly.

----------------------------------------------------------------------------------------------------
prior to calculating the anaylsis we convert everything to USD so that we have consistent statistics using the following exchange rates:

        'USD': 1.0,
        'EUR': 1.1637,  # EURUSD = 1.1637
        'GBP': 1.3623,  # GBPUSD = 1.3623
        'CHF': 1.2421,  # CHFUSD = 1.2421
        'JPY': 1/145.41  # USDJPY = 145.41, so 1 JPY = 1/145.41 USD

----------------------------------------------------------------------------------------------------------
Task #2.3

1) If I had to build it for production so I will use more robust 
 1.1) Use FX library to convert amount at the time of the transaction. 
 1.2) Use dateutil for better timezone handling.
 1.3) Implement better monitoring.
 1.4) Use pytest and write tests.
 1.5) Designa scalable architecture using Kafka. Potentially use Kafka streams and process transaction in real time.
 1.6) Upload the data to a structured database
 1.7) Build a UI to allow support staff to fix exceptions
 1.8) Use LLM to fix transactions exceptions
 1.9) User Docker and Kubernetes to enable scaling
  
