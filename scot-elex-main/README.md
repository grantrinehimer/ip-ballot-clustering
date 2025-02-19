# Scottish Election Information 

This is a repository containing the data from Scottish local government elections
conducted by ranked voting in the years 2007, 2012, 2017, and 2022. This data
was originally assembled and cleaned thanks to the work of [David McCune](https://www.jewell.edu/faculty/david-mccune), 
who collected cast vote records from Scottish sources to create 1100 files in 
a file format called BLT format.  (Originals can be found in the `blt_files`
folder here.)

In this repo, we have set aside 30 by-elections (also known as "special elections")
because those do not include candidates' party labels, leaving 1070 ranked elections
with party labels.

Scottish council elections are conducted using voting rule known as Single
Transferable Vote (STV) which sets a threshold for election related to the number of 
seats to be filled; candidates are then elected when they pass the threshold, or 
eliminated if nobody passes the threshold, with surplus votes transfering until enough
candidates are elected.

## How to read the CSV files

The CSV files compiled here follow a similar format to the BLT files from which they were
derived. Here is a simple synopsis of the data contained in each of the rows of the file.

- **Row 1:** Number of candidates, Number of seats
- **Rows of ballot frequencies:** 
  - The first entry of the row is the number of voters who chose a particular ballot
  - All numbers in the row following the first number describe the ordering of the ballot,
    so a row with the description `12,3,1,2` says that there were 12 ballots with the
    preference order 3>1>2
- **Rows following ballot frequencies:** 
  Immediately following the ballots is an ordered list of candidates: 
  Candidate1, Candidate2, ...
- **Last row:** Location
  The last line of the file contains the name of the Ward where the election was held.
  To see the year, consult the file name.

### Example 

Here is an example CSV file.

```
3,2,
12,1,
88,1,3,2,
14,2,3,
8,3,
"Candidate 1","Alvin the First","Party A (A)",
"Candidate 2","Becca the Second","Party B (B)",
"Candidate 3","Carrie the Third","Party C (C)",
"Wardy McWard Town",
```

How to read it:

```
3,2,                                                # There are 3 candidates running for 2 seats
12,1,                                               # 12 Bullet votes for candidate 1 were cast
88,1,3,2,                                           # 88 Ballots ranking 1>3>2 were cast
14,2,3,
8,3,
"Candidate 1","Alvin the First","Party A (A)",      # The name of candidate 1 and their party
"Candidate 2","Becca the Second","Party B (B)",
"Candidate 3","Carrie the Third","Party C (C)",
"Wardy McWard Town",                                # The name of the ward
```

## List of Parties in 2007, 2012, 2017, and 2022 Elections

- Alba Party for Independence (API)
- Borders (Borders)
- Britannica Party (BP)
- British National Party (BNP)
- British Unionists (BU)
- Christian Peoples Alliance (CPA)
- Communist Party of Britain (Comm)
- (Scottish) Conservative and Unionist Party (Con)
- Cumbernauld Independent Councillors Alliance (CICA)
- East Dunbartonshire Independent Alliance (EDIA)
- East Kilbride Alliance (EKA)
- Freedom Alliance (FA)
- Glasgow First (Glasgow First)
- (Scottish) Green (Gr)
- Independence for Scotland Party (ISP)
- Independent (Ind)
- Independent Alliance North Lanarkshire (IANL)
- Labour (Lab)
- Labour and Co-operative Party (LabCo)
- Liberal (Lib)
- Liberal Democrat (LD)
- Libertarian (Libtn)
- Monster Raving Loony (MVR)
- National Front (NF)
- No Referendum, Maintain Union, Pro-Brexit (NRMUPB)
- Orkney Manifesto Group (OMG)
- Piarate (Pir)
- RISE (RISE)
- Rubbish (Rubbish)
- Scotland Independent Network (ScIN)
- Scottish Christian (SC)
- Scottish Eco-Federalist Party (SEFP)
- Scottish Family Party (SFP)
- Scottish National Party (SNP)
- Scottish Senior Citizens (SSC)
- Scottish Unionist (SU)
- Social Democratic Party (SDP)
- Socialist (Soc)
- Socialist Labour Party (SLP)
- Solidarity (Sol)
- Sovereignty (Sov)
- The Pensioner's Party (TPP)
- Trade Unionist and Socialist Coalition (TUSC)
- UK Independence Party (UKIP)
- Vanguard Party (Van)
- Volt UK (Volt)
- West Dunbartonshire Community (WDuns)
- Women's Equality Party (WEP)
- Worker Party of Britain (WPB)
