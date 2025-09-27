#==== SI Fig 2a =================================================================================
# Read Data 
Primary_tumor_diameter <- read.csv("data/Primary_tumor_diameter.csv")
Primary_tumor_diameter$Day <- as.factor(Primary_tumor_diameter$Day)

# ANOVA 
AOV <- aov(Primary_tumor_diameter ~ Day, data = Primary_tumor_diameter)
print(summary(AOV), digits = 15)

# Tukey's post hoc test
print(TukeyHSD(AOV), digits = 15)





#==== SI Fig 2b =================================================================================
# Read Data 
POLN_weight <- read.csv("data/POLN_weight.csv")
POLN_weight$Day <- as.factor(POLN_weight$Day)

# ANOVA 
AOV <- aov(POLN_weight ~ Day, data = POLN_weight)
print(summary(AOV), digits = 15)

# Tukey's post hoc test
print(TukeyHSD(AOV), digits = 15)