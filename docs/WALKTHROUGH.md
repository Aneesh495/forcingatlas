# Walkthrough

This is the path through the local files and the atlas. It uses the acquired tables in `data/raw` and the bundle written to `web/public/bundle`.

```bash
python -m forcing_atlas.cli walkthrough
python -m forcing_atlas.cli export-atlas
```

Then open the atlas and leave the default layers on.

1. HadCRUT5 analysis global and Berkeley Earth air-above-sea-ice are both on the chart. Berkeley has been shifted from 1951-1980 onto 1961-1990. The ocean parts share HadSST, so the comparison is not two independent ocean records.
2. PAGES 2k is the violet line and carries a proxy badge. Years before the instrumental window are kept in the file and are not drawn as thermometers. The chart's horizontal scale follows the instrumental and emulation series so the Common Era does not squash 1850-2023.
3. The dashed brown line is the two-box model with the full IGCC effective radiative forcing. The dashed green line uses only solar and volcanic forcing and the same parameters. They are different questions. The natural-only line is not a `hist-nat` experiment.
4. Click a 5 degree cell on the flat map. The side panel reports the anomaly when the non-infilled grid has one, and says the cell is empty when it does not. Coverage is the fraction of that year's days present.
5. Move to 2100. The map states that the regional layer is unavailable. The stabilize and decline series are global emulations under declared forcing paths. They are not SSPs, and the map is not filled by repeating the last year.

The holdout printed by the walkthrough is 1995 onward. Persistence, the linear trend, and the two-box model are scored on that window. The fit does not see it.
