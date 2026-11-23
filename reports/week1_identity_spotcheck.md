# Independent audit of speaker identities and dated affiliations

**Result.** All ten frozen speeches in a stratified spot-check agreed with public congressional biographies or directories. For each Congress and chamber, we selected the row with the lexicographically smallest SHA-256 of the UTF-8 string `week1-identity-qa|` followed by its edition-qualified speech ID. We independently read the speech ID, original speaker name, seat and date from the Stanford daily description and SpeakerMap files, then checked the named legislator's period of service and affiliation against the sources in the table. The corpus SHA-256 for this check was `02649f7e5c7a39bb91f071a09d91a840214b9d8f6607131ae1f3ff1d09198db9`. Ten cases give coverage across the ten sampling strata. They do not estimate the population linkage error rate.

| Congress | Chamber | Daily speech ID | Date | Identity | BioGuide | Label | Official reference |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 110 | House | 1100141157 | 2008-03-05 | Ike Skelton, MO | S000465 | D | [House Historian](https://history.house.gov/People/Listing/S/SKELTON%2C-Isaac-Newton-%28Ike%29%2C-IV-%28S000465%29/) |
| 110 | Senate | 1100027079 | 2007-03-22 | Kent Conrad, ND | C000705 | D | [Congressional Directory](https://www.govinfo.gov/content/pkg/CDIR-2007-08-09/html/CDIR-2007-08-09-ND-S-1.htm) |
| 111 | House | 1110089376 | 2010-03-16 | Todd Platts, PA | P000585 | R | [House Historian](https://history.house.gov/People/Detail/20009) |
| 111 | Senate | 1110142418 | 2010-11-17 | Sherrod Brown, OH | B000944 | D | [House Historian](https://history.house.gov/people/detail/9996) |
| 112 | House | 1120023563 | 2011-04-13 | Jared Polis, CO | P000598 | D | [House Historian](https://history.house.gov/People/Detail/20034) |
| 112 | Senate | 1120038379 | 2011-06-22 | Rob Portman, OH | P000449 | R | [House Historian](https://history.house.gov/People/Detail/19838) |
| 113 | House | 1130041408 | 2013-07-25 | John Shimkus, IL | S000364 | R | [House Historian](https://history.house.gov/People/Listing/S/SHIMKUS%2C-John-M--%28S000364%29/) |
| 113 | Senate | 1130068175 | 2014-01-08 | Richard Durbin, IL | D000563 | D | [Congressional Directory](https://www.govinfo.gov/content/pkg/CDIR-2014-02-18/html/CDIR-2014-02-18-IL-S-1.htm) |
| 114 | House | 1140034532 | 2015-06-11 | Paul Ryan, WI | R000570 | R | [House Historian](https://history.house.gov/People/Detail/20785) |
| 114 | Senate | 1140013577 | 2015-03-11 | Bernard Sanders, VT | S000033 | I | [House Historian](https://history.house.gov/People/Listing/S/SANDERS%2C-Bernard-%28S000033%29/) |

In the Polis case the Stanford description spells the speaker's surname `POLLS`. The official April 13 transcript attributes the same remarks on defense spending to Polis, independently confirming the SpeakerMap correction. [Congressional Record](https://www.govinfo.gov/content/pkg/CREC-2011-04-13/pdf/CREC-2011-04-13-house.pdf)

## Switchers and minority parties

We separately checked all **404** daily SpeakerMap rows for Parker Griffith and Arlen Specter in Congress 111 that joined a description in the same chamber. Expected labels were determined from published switch dates. This test covers the source pool before sampling.

| Legislator | Published transition and treatment | Observed outcomes |
| --- | --- | --- |
| Parker Griffith, G000557 | Democrat before 2009-12-22, Republican from that date. [House Historian](https://history.house.gov/People/Detail/14273) | 14 D, 15 R. The archive assigns R to all 29. Examples on opposite sides are 1110072450 and 1110077632. |
| Arlen Specter, S000709 | Public switch announced 2009-04-28, formally recorded by the Senate on 2009-04-30. The resolver withholds labels on April 28–29. [Senate committee chronology](https://www.govinfo.gov/content/pkg/CPRT-111SPRT65788/pdf/CPRT-111SPRT65788.pdf) · [Senate notification](https://www.govinfo.gov/content/pkg/CRECB-2009-pt9/pdf/CRECB-2009-pt9-Pg11277.pdf) | 133 R, 241 D, one excluded (1110034319 on April 28). The archive assigns D to all 375. Examples on opposite sides are 1110033604 and 1110041423. |

Date-aware labeling changes 147 of these source-period labels and excludes one transition record. Among the final 20,000 speeches, Congress 111 contributes 11 R and 13 D rows for Specter and two R rows for Griffith. The raw audit includes both sides of both switches even where the sample selected only one side.

The final descriptive labels cover Sanders and Angus King as Independents (I) and Joseph Lieberman as Independent Democrat (ID). Congress 114's 29 House records coded I in the Stanford map belong to nonvoting delegate Gregorio Sablan. The House Historian dates his change to Democrat to 2009. His records are excluded by the voting-member cohort rule. [Senate affiliations](https://www.senate.gov/senators/SenatorsRepresentingThirdorMinorParties.htm) · [House Historian](https://history.house.gov/People/Listing/S/SABLAN%2C-Gregorio-Kilili-Camacho-%28S001177%29/)
