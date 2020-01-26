# capstone-project
Includes the code used in my capstone project. The paper goes into details for reasoning behind my decisions.

GOAL

Use characteristics of pitches for predicting pitcher success.

ABSTRACT

Over the last few years, some of the more analytically inclined Major League Baseball
organizations will acquire a pitcher and help the pitcher perform much better than they
had in the past. The Houston Astros are a prime example of doing this multiple times.
Their first well-known example of this was when the Astros signed Charlie Morton, who
was a below-average pitcher over his 9-year career, and helped him become a dominant,
All-Star pitcher. Part of the solution was the Astros showing and convincing Morton
which pitches he has that are good and which ones are bad. Morton changed his pitch
usage to throw his good pitches more and bad pitches less resulting in a complete
turnaround of his career.
This paper attempts to predict pitcher performance based on the quality of the pitches a
pitcher throws. Statcast pitching data is used as the input features for predicting xwOBA
based on the 2019 season. Models are created based on two different sets of input
features. One set of features includes all pitchers that threw at least 100 pitches during the
season, and the other set of features only includes pitchers that meet the requirements for
being on Statcast Pitch Movement Leaderboard: “pitcher must have 3 pitches thrown per
team game” and “must use it at least 5% of the time” (Baseball Savant, n.d.-a).
Two machine learning methods were used for predicting xwOBA. Random Forest
Regression was used for the first set of features, and Random Forest Regression and
Linear Regression were used for the second set of features. Linear Regression resulted in
the highest accuracy with a 91.80% Mean Average Percentage Error.



If you would like to view the entire paper, send me an email at connercapdau@gmail.com.
