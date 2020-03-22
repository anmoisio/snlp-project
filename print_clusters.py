import os

file_path = os.path.join("clusters", "wikipedia_clusters_full_k20.txt")

with open(file_path, "r") as f:
    clusters = f.read().splitlines()

clustered = [line.split() for line in clusters]

# separate clusters into different lists
n_clusters = 20
clusters = [[] for i in range(n_clusters)]
for line in clustered[1:]:
    clusters[int(line[1])].append(line[0])

for c in clusters:
    print(c[:50]) # print 50 words from each cluster
