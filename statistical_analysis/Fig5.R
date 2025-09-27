#==== Read Data ================================================================
Pdot_total_data <- read.csv("data/Accumulated_Pdot_amount.csv")

#==== Hotteling's T2 test ======================================================
# Install the package
install.packages("Hotelling")

# Load the package
library(Hotelling)

#Data Prepossessing
day1 <- subset(Pdot_total_data, Day == "1", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day5 <- subset(Pdot_total_data, Day == "5", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day9 <- subset(Pdot_total_data, Day == "9", select = c("F8BT", "MEHPPV", "TPP_PFO"))


# analysis
hotelling_day1_5_pval <- hotelling.test(x = day1, y = day5)$pval
hotelling_day5_9_pval <- hotelling.test(x = day5, y = day9)$pval
hotelling_day9_12_pval <- hotelling.test(x = day9, y = day12)$pval


hotteling_results <- c(
  hotelling_day1_5_pval,
  hotelling_day5_9_pval,
  hotelling_day9_12_pval
)

# Create a character vector for column names
names_vector <- c("0_vs_1", "1_vs_5", "5_vs_9", "9_vs_12", "0_vs_12")

# Assign names to the vector elements
names(hotteling_results) <- names_vector

# Print the final matrix
print(hotteling_results)