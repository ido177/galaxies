import pandas as pd
import numpy as np
import scipy as sp


def main():
    df = pd.read_csv('/Users/cyrilromans/Downloads/groups.tsv', delimiter='\t').dropna()

    igl_mean_1 = df['mean_mu'][df.features == 1].mean()
    igl_mean_0 = df['mean_mu'][df.features == 0].mean()

    swn_igl_mean_1 = sp.stats.shapiro(igl_mean_1)  #df['mean_mu'][df.features == 1])
    swn_igl_mean_0 = sp.stats.shapiro(igl_mean_0) #df['mean_mu'][df.features == 0])

    fligner_test = sp.stats.fligner(df['mean_mu'][df.features == 1], df['mean_mu'][df.features == 0])

    anova_test = sp.stats.f_oneway(swn_igl_mean_1, swn_igl_mean_0)

    print(round(swn_igl_mean_1.pvalue, 5), round(swn_igl_mean_0.pvalue, 5), round(fligner_test.pvalue, 5), round(anova_test.pvalue, 5))


if __name__ == '__main__':
    main()
