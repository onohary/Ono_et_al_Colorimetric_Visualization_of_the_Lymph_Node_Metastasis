#==== Read Data ================================================================
Pdot_diameter <- read.csv("data/Pdot_diameter.csv")

#==== ANOVA ====================================================================
AOV <- aov(diameter ~ Pdot_Type, data = Pdot_diameter)
print(summary(AOV), digits = 15)

#==== Tukey's post hoc test ====================================================
print(TukeyHSD(AOV), digits = 15)
