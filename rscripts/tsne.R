library(Rtsne)

dest <- read.csv('~/Projects/ean/data/destinations.csv')

n <- 10000
data <- dest[1:n, -1]
labels <- dest[1:n, 1]
dupl <- duplicated(data)
td <- Rtsne(data[!dupl, ], k = 2, verbose = T)

plot(td$Y, main="tsne")

#colors = rainbow(sum(!dupl))
#names(colors) <- labels[!dupl]
#plot(td$Y, t='n', main="tsne")
#text(td$Y, labels=labels[!dupl], col=colors[labels[!dupl]])