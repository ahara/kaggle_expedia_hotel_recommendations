library(data.table)
library(ggplot2)


# is_mobile
train <- fread('~/Projects/ean/data/train.csv', select = c('is_mobile', 'hotel_cluster'))
summary(train)

ggplot(train, aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)

ggplot(subset(train, train$is_mobile == 0), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$is_mobile == 1), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)


# is_package and channel
train <- fread('~/Projects/ean/data/train.csv', select = c('is_package', 'channel', 'hotel_cluster'))
summary(train)
table(train$channel)

ggplot(subset(train, train$is_package == 0), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$is_package == 1), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)

ggplot(subset(train, train$channel == 6), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$channel == 10), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)


# srch_adults_cnt, srch_children_cnt and srch_rm_cnt
train <- fread('~/Projects/ean/data/train.csv',
               select = c('srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'hotel_cluster'))
summary(train)
ggplot(subset(train, train$srch_adults_cnt > 2), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$srch_adults_cnt == 2), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$srch_adults_cnt < 2), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)

ggplot(subset(train, train$srch_children_cnt == 0), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$srch_children_cnt > 0), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)

ggplot(subset(train, train$srch_children_cnt <= 1), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$srch_children_cnt == 1), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)
ggplot(subset(train, train$srch_rm_cnt > 1), aes(x = hotel_cluster)) + geom_histogram(binwidth = 1)

# srch_destination_type_id
train <- fread('~/Projects/ean/data/train.csv', select = c('srch_destination_type_id', 'hotel_cluster'))
summary(train)

ggplot(train, aes(x = hotel_cluster)) + geom_histogram(binwidth = 1) + facet_grid(srch_destination_type_id ~ ., scales = 'free')


#
train <- fread('~/Projects/ean/data/train.csv', select = c('srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster'))
