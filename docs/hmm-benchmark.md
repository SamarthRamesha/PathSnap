# HMM benchmark on complete training trips

The benchmark reads the GPS track and directed network from the raw files. It
projects the trip into a local metric frame, finds the eight nearest valid arcs
for each GPS observation, and scores them with the decoder's Gaussian distance
emission. It keeps original arc IDs when filtering zero-length arcs. Every
returned segment is checked for directed adjacency before the map is written.

The benchmark measures graph construction and Viterbi together. It reports
raw input loading and candidate generation separately, and excludes map
rendering and HTTP overhead from both figures.

## Reproduce

From the root of the `feature/hmm-ui` checkout, fetch the dataset on `main`
into a separate worktree and run the script:

```sh
git fetch origin main
git worktree add --detach ../PathSnap-data FETCH_HEAD
python -m pip install "numpy>=1.26" "folium>=0.16"
python -m demos.benchmark_trip ../PathSnap-data/data/train/00000039 --output real-trip-00000039.html
python -m demos.benchmark_trip ../PathSnap-data/data/train/00000010 --output real-trip-00000010.html
```

Open the HTML in a browser with internet access for the basemap and map assets.
Use the layer controls to toggle GPS observations, route segments and decoder
break markers. The script prints timing and route diagnostics as JSON.

## Results

Measured in the Linux development environment with Python 3.12.14 on an AMD
EPYC 9V74 host. Each trip used eight candidates per observation and unchanged
decoder defaults. Candidate selection for this benchmark uses straight-line
arc geometry and distance only.

The raw data comes from `main` commit `d542ce1`. Reference routes were not
used to choose or score candidates. These results measure runtime,
coverage, and internal connectivity. Accuracy against held-out reference
routes requires the model's per-observation probabilities and the agreed
evaluation metrics.

Download a generated HTML file from GitHub and open it locally to view
the map.

| Trip | GPS rows matched | Directed arcs | Prep | Decoder runs | Median decode | Route segments | Breaks |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| [00000039](maps/real-trip-00000039.html) | 363 / 363 | 2,388 | 0.067 s | 0.0254, 0.0258, 0.0461 s | 0.026 s | 1 segment, 172 arcs | 0 |
| [00000010](maps/real-trip-00000010.html) | 2,711 / 2,711 | 15,715 | 4.125 s | 35.4127, 34.0228, 36.0723 s | 35.413 s | 1 segment, 5,399 arcs | 0 |

I stopped the extra stress run on trip `00000001` after about 5 minutes
28 seconds. The three-run benchmark had yet to produce a timing summary or
map. That trip has 1,315 GPS rows and 165,790 raw directed arcs. The elapsed
time is for the unfinished benchmark process, so it does not establish a
per-run decode time. The 35-second median on the completed larger trip
already makes decoder latency material for synchronous API requests.
