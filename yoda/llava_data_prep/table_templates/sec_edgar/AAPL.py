synth_tables = []

# Template 1: AAPL_Revenue_simple_columns
ex1 = {"columns": "\\begin{tabular}{| l c c |}\n\\hline\n",

"tsv_formatted": """\tThree Months Ended\t
\tDecember 31,\tDecember 25
\t2022\t2021
iPhone(1)\t$**65,775\t$**71,628
Mac(1)\t***7,735\t***10,852
iPad(1)\t***9,396\t***7,248
Wearables, Home and Accessories\t***13,482\t***14,701
Services(3)\t***20,766\t***19,516
*Total net sales\t**117,154\t**123,945
"""}

synth_tables.append(ex1)

# Template 2: AAPL_cash_market_secs
ex2 = {"columns": "\\begin{tabular}{| l c c c c c c c |}\n\\hline\n",

"tsv_formatted": """\t\t\t\t\tCash and\tCurrent\tNon-Current
\tAdjusted\tUnrealized\tUnrealized\tFair\tCash\tMarketable\tMarketable
\tCost\tGains\tLosses\tValue\tEquivalents\tSecurities\tSecurities
\\hline\t\t\t\t\t\t\t
Cash\t$**18,546\t$***-\t$***-\t$**18,546\t$**18,546\t$***-\t$***-
Level 1:\t\t\t\t\t\t\t
*Money and market funds***\t***2,929\t****-\t****-\t***2,929\t***2,929\t****-\t****-
*Mutual funds\t****274\t****-\t****(47)\t****227\t****-\t****227\t****-
\\hline\t\t\t\t\t\t\t
**Subtotal***\t***3,203\t****-\t****(47)\t***3,156\t***2,929\t****227\t****-
\\hline\t\t\t\t\t\t\t
Level 2:\t\t\t\t\t\t\t
*U.S. Treasury securities\t**25,134\t****-\t***(1,725)\t**23,409\t****338\t***5,091\t**17,980
*U.S. agency securities\t***5,823\t****-\t****(655)\t****5,168\t****-\t****240\t***4,928
*Non-U.S. government securities\t**16,948\t****2\t***(1,201)\t**15,749\t****-\t***8,806\t***6,943
*Certificates of deposit and time deposits\t**87,148\t****9\t***(7,707)\t**79,450\t****-\t***9,023\t**70,427
*Commerical paper\t****718\t****-\t****-\t****718\t****28\t****690\t****-
*Corporate debt securities\t**87,148\t****9\t***(7,707)\t**79,450\t****-\t***9,023\t**70,427
*Municipal securities\t****921\t****-\t****(35)\t****886\t****-\t****266\t****620
*Mortgage- and asset-backed securities\t**22,553\t****-\t***(2,593)\t**19,960\t****-\t****53\t**19,907
\\hline\t\t\t\t\t\t\t
**Subtotal\t*161,312\t****11\t**(13,916)\t*147,407\t***2,171\t**24,431\t*120,805
\\hline\t\t\t\t\t\t\t
***Total\t$*183,061\t$****11\t$**(13,963)\t$*169,109\t$**23,646\t$**24,658\t$*120,805
\\hline\t\t\t\t\t\t\t
"""}

synth_tables.append(ex2)

# Template 3: AAPL_Country_Revenues
ex3 = {"columns": "\\begin{tabular}{| l c c |}\n\\hline\n",
 
"tsv_formatted": """\t*****Three Months\t
\tDecember 31,\tDecember 25,
\t2022\t2021
Americas:*********\t\t
*Net sales\t$**49,278\t$**51,496
*Operating income\t$**17,864\t$**19,585
\t\t\t
Europe:*********\t\t
*Net sales\t$**27,681\t$**29,749
*Operating income\t$**10,017\t$**11,545
\t\t\t
Greater China:*********\t\t
*Net sales\t$**23,905\t$**25,783
*Operating income\t$**10,437\t$**11,183
\t\t\t
Japan:*********\t\t
*Net sales\t$**6,755\t$**7,107
*Operating income\t$**3,236\t$**3,349
\t\t\t
Rest of Asia Pacific:*********\t\t
*Net sales\t$**9,535\t$**9,810
*Operating income\t$**3,851\t$**3,995
"""}

synth_tables.append(ex3)

# Template 4: AAPL_XML1
ex4 = {"columns": "\\begin{tabular}{| >{\\color{blue}}l c c |}\n\\hline\n",
 
 "tsv_formatted": """Diluted (in dollars per share)\t$*1.88\t$*2.10
\\textbf{Shares used in computing earnings per share:}\t\t
Basic (in shares)\t15,892,723\t16,391,724
Dilutes (in shares)\t15,955,718\t16,519,291
Products\t\t
Net sales\t*$*96,388\t*$*104,429
Cost of sales\t60,765\t64,309
Service\t\t
Net sales\t20,766\t19,516
Cost of sales\t*$*6,057\t*$*5,393
"""}

synth_tables.append(ex4)

# Template 5: AAPL_Condensed_Consolidated_Balance_Sheets - Became too large, incomplete rows
ex5 = {"columns": "\\begin{tabular}{| l c c c |}\n\\hline\n",
 
 "tsv_formatted": """\t\tApril 1,\tSeptember 24,
\t\t2023\t2022
\\hline\t\t\t
\t\\textbf{ASSETS:}***********\t\t
Current assests:***********\t\t\t
*Case and cash equivalents:***********\t\t$****24,687\t$****23,646
*Marketable securities:***********\t\t*****31,185\t*****24,658
*Accounts receivable, net:***********\t\t*****17,936\t*****28,184
*Inventories:***********\t\t******7,482\t******4,946
*Vendor non-trade receivables:***********\t\t*****17,963\t*****32,748
*Other current assets***********\t\t*****13,660\t*****21,223
\\hline\t\t\t
**Total current assets***********\t\t*****112,913\t*****135,405
\t\t\t
Non-current assets:***********\t\t\t
*Marketable securities***********\t\t*****110,461\t*****120,805
*Property, plant and equipment, net***********\t\t******43,461\t******42,117
*Other non-current assets***********\t\t*****65,388\t*****54,428
\\hline\t\t\t
**Total non-current assets***********\t\t*****219,247\t*****217,350
\\hline\t\t\t
***Total assets***********\t\t$****332,160\t$****352,755
\\hline\t\t\t
\t\\textbf{LIABILITIES AND SHAREHOLDERS' EQUITY:}****\t\t
Current liabilities:***********\t\t\t
*Accounts payable:***********\t\t$****42,945\t$****64,115
*Other current liabilities:***********\t\t*****56,345\t*****60,845
*Deferred revenue***********\t\t******8,131\t******7,912
*Commercial papers***********\t\t******1,996\t******9,982
*Term debt***********\t\t*****10,578\t*****11,128
\\hline\t\t\t
**Total current liabilities***********\t\t*****120,075\t*****153,982
\\hline\t\t\t
Non-current liabilities:***********\t\t\t
*Term debt***********\t\t*****97,041\t*****98,959
*Other non-current liabilities***********\t\t*****52,886\t*****49,142
**Total non-current liabilities***********\t\t*****149,927\t*****148,101
\\hline\t\t\t
***Total liabilities***********\t\t*****270,002\t*****302,083
"""}

synth_tables.append(ex5)

# Template AAPL_Condensed_Consolidated_Financial_Statements
ex6 = {"columns": "\\begin{tabular}{| l c c |}\n\\hline\n",

"tsv_formatted": """\\textbf{Inventories}\t\t
\t\\textbf{April 1,}\t\\textbf{September 24,}
\t\\textbf{2023}\t\\textbf{2022}
\\hline\t\t
Components********************\t$*****3,379\t$*****1,637
Finished goods******************\t$*****4,103\t$*****3,309
\\hline\t\t
*Total inventories**************\t$*****7,482\t$*****4,946
\t\t
\t\t
\\textbf{Property, Plant, and Equipment, Net}*******\t\t
\t\t
\t\\textbf{April 1,}\t\\textbf{September 24,}
\t\\textbf{2023}\t\\textbf{2022}
Gross property, plant and equipment*********\t$****113,006\t$****114,457
Accumulated deprecation and amorization**************\t****(69,668)\t****(72,340)
\\hline\t\t
*Total property, plant, and equipment, net**********\t$****43,398\t$****42,117
"""}

synth_tables.append(ex6)

# Template APPL_Products_and_Services_Performance

ex7 = {"columns": "\\begin{tabular}{| l c c r c c r |}\n\\hline\n",
 
"tsv_formatted": """\\textbf{Products and Services Performance}\t\t\t\t\t\t
\t\t\\textbf{Three Months Ended}\t\t\\textbf{Six Months Ended}\t\t
\\hline\t\t\t\t\t\t
****\t\\textbf{April 1,}\t\\textbf{March 26,}\t\t\\textbf{April 1,}\t\\textbf{March 26,}\t
****\t\\textbf{2023}\t\\textbf{2022}\t\\textbf{Change}\t\\textbf{2023}\t\\textbf{2022}\t\\textbf{Change}
\\hline\t\t\t\t\t\t
Net sales by category\t\t\t\t\t\t
*iPhone***\t$***51,334\t$***50,570\t*****2%\t$**117,109\t$**122,198\t****(4)%
*Mac***\t*****7,168\t****10,435\t****(31)%\t****14,903\t****21,287\t***(30)%
*iPad***\t*****6,670\t****7,646\t****(13)%\t***16,066\t***14,894\t*****8%
*Wearables, Home and Accessories\t*****8,757\t****8,806\t*****(1)%\t***22,239\t***23,507\t***(5)%
*Services***\t****20,907\t***19,821\t*******5%\t***41,673\t***39,337\t****6%
\\hline\t\t\t\t\t\t
**Total new sales**\t$***94,836\t$***97,278\t*****(3)%\t$**211,990\t$**221,223\t*****(4)%
"""}

synth_tables.append(ex7)

