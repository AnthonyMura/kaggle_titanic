| Variable    | Definition                    | Key / Values                                  |
|:------------|:------------------------------|:----------------------------------------------|
| `survival`  | Survival status               | `0` = No, `1` = Yes                           |
| `pclass`    | Ticket class                  | `1` = 1st, `2` = 2nd, `3` = 3rd              |
| `sex`       | Sex                           | `male`, `female`                              |
| `age`       | Age in years                  | Numeric value (fractional if < 1 year)        |
| `sibsp`     | # of siblings / spouses       | Integer count aboard the Titanic              |
| `parch`     | # of parents / children       | Integer count aboard the Titanic              |
| `ticket`    | Ticket number                 | Alphanumeric string                           |
| `fare`      | Passenger fare                | Numeric value (British Pound)                 |
| `cabin`     | Cabin number                  | Alphanumeric string (e.g., "C123")            |
| `embarked`  | Port of Embarkation           | `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |

Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.