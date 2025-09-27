#==== SI Fig 5a =================================================================================
# Read Data 
Pdot_total_data <- read.csv("data/Accumulated_Pdot_amount.csv")

# Hotteling's T2 test 
# Install the package
install.packages("Hotelling")

# Load the package
library(Hotelling)

#Data Prepossessing
Normal <- subset(Pdot_total_data, Day == "0", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day1 <- subset(Pdot_total_data, Day == "1", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day5 <- subset(Pdot_total_data, Day == "5", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day9 <- subset(Pdot_total_data, Day == "9", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day12 <- subset(Pdot_total_data, Day == "12", select = c("F8BT", "MEHPPV", "TPP_PFO"))


# analysis
hotelling_Normal_day1_pval <- hotelling.test(x = Normal, y = day1)$pval
hotelling_day1_5_pval <- hotelling.test(x = day1, y = day5)$pval
hotelling_day5_9_pval <- hotelling.test(x = day5, y = day9)$pval
hotelling_day9_12_pval <- hotelling.test(x = day9, y = day12)$pval

hotteling_results <- c(
  hotelling_Normal_day1_pval,
  hotelling_day1_5_pval,
  hotelling_day5_9_pval,
  hotelling_day9_12_pval
)

# Create a character vector for column names
names_vector <- c("Normal_vs_1", "1_vs_5", "5_vs_9", "9_vs_12")

# Assign names to the vector elements
names(hotteling_results) <- names_vector

# Print the final matrix
print(hotteling_results)

#==== SI Fig 5b =================================================================================
#Data Prepossessing
Pdot_total_data$Day <- as.factor(Pdot_total_data$Day)
F8BT <- Pdot_total_data[, c("F8BT", "Day")]

# ANOVA 
AOV_Pdot_accumulation_F8BT <- aov(F8BT ~ Day, data = F8BT)
print(summary(AOV_Pdot_accumulation_F8BT), digits = 15)

# Tukey's post hoc test
print(TukeyHSD(AOV_Pdot_accumulation_F8BT), digits = 15)

#==== SI Fig 5c =================================================================================
#Data Prepossessing
MEHPPV <- Pdot_total_data[, c("MEHPPV", "Day")]

# ANOVA 
AOV_Pdot_accumulation_MEHPPV <- aov(MEHPPV ~ Day, data = MEHPPV)
print(summary(AOV_Pdot_accumulation_MEHPPV), digits = 15)

# Tukey's post hoc test
print(TukeyHSD(AOV_Pdot_accumulation_MEHPPV), digits = 15)

#==== SI Fig 5d =================================================================================
#Data Prepossessing
TPP_PFO <- Pdot_total_data[, c("TPP_PFO", "Day")]

# ANOVA 
AOV_Pdot_accumulation_TPP_PFO <- aov(TPP_PFO ~ Day, data = TPP_PFO)
print(summary(AOV_Pdot_accumulation_TPP_PFO), digits = 15)