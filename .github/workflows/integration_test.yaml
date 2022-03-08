name: integration tests
on: [push]
jobs:
  run:
    runs-on: [ubuntu-latest]
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          auto-update-conda: false
          activate-environment: quant
          environment-file: environment.yaml
          python-version: 3.9
      - uses: iterative/setup-cml@v1
      - name: Run integration tests
        shell: bash -l {0}
        run: |
          python run_test_cases.py
      - name: Write CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          cat output/results.csv >> report.md
          cml-send-comment output/results.csv
