import pandas as pd
import torch
from torch.utils.data import Dataset # DataLoader, Subset

class CountDataset(Dataset):
    def __init__(self,
                 gene_expression,
                 cat_df=None, cont_df=None, treatment_df=None, # treatment_df is an unused parameter
                 cat_cols=None, cont_cols=None, norm=None,
                 H_dim=1, data_dist='gaussian'
        ):
        """
        gene_expression: numpy array or tensor (n_cells, n_genes)
        cat_df: pandas DataFrame with categorical columns, optional
        cont_df: pandas DataFrame with continuous columns, optional
        treatment_df: optional treatment sequence DataFrame
        cat_cols / cont_cols: optional subset of columns to use
        data_dist: 'gaussian' or 'counts'
        """
        self.gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        self.n_samples = self.gene_expression.shape[0]
        self.library_size = self.gene_expression.sum(1).reshape(self.n_samples, 1)
        self.aux_info = torch.empty_like(self.library_size)

        self.cat_df = cat_df
        self.cont_df = cont_df
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.H_dim = H_dim
        self.data_dist = data_dist

        # only for Gaussian case
        if norm=='log1p':
            print('Applying log1p transformation to gene expression')
            self.gene_expression = torch.log1p(self.gene_expression)

        # Handle categorical
        if self.cat_df is not None:
            if self.cat_cols is None:
                self.cat_cols = list(cat_df.columns)
            
            self.cat_cols_factor = [f'{col}_factor' for col in self.cat_cols]
            cat_df_ = cat_df.copy()
            for col in self.cat_cols:
                cat_df_[f'{col}_factor'] = pd.Categorical(cat_df_[col]).codes
            self.cat_df_tensor = torch.tensor(cat_df_[self.cat_cols_factor].values, dtype=torch.int32)
        else:
            self.cat_df_tensor = None

        # Handle continuous
        if self.cont_df is not None:
            if self.cont_cols is not None:
                self.cont_df_tensor = torch.tensor(cont_df[self.cont_cols].values, dtype=torch.float32)
            else:
                self.cont_cols = list(cont_df.columns)
                self.cont_df_tensor = torch.tensor(cont_df.values, dtype=torch.float32)
        else:
            self.cont_df_tensor = None

        self.treatment_df = None # unused parameter

        # set dimensions
        self.data_dim = self.gene_expression.shape[1]
        self.treatment_dim = 0
        self.cat_dim = self.cat_df_tensor.shape[1] if self.cat_df_tensor is not None else 0
        self.num_cats = [int(max(self.cat_df_tensor[:,i])+1) for i in range(self.cat_dim)] if self.cat_df_tensor is not None else [0]
        self.cont_dim = self.cont_df_tensor.shape[1] if self.cont_df_tensor is not None else 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_genes = self.gene_expression[idx]

        # Categorical
        if self.cat_df_tensor is not None:
            # u_cat = {col: torch.tensor(self.cat_df[col].iloc[idx], dtype=torch.float32) for col in self.cat_df.columns}
            u_cat = self.cat_df_tensor[idx]
        else:
            u_cat = torch.empty(0)

        # Continuous
        if self.cont_df_tensor is not None:
            u_cont = self.cont_df_tensor[idx]
        else:
            u_cont = torch.empty(0)

        # return values when counts
        if self.data_dist == 'counts':
            aux_info = torch.empty(0)
            library_size = self.library_size[idx]

            return x_genes, u_cat, u_cont, aux_info, library_size

        return x_genes, u_cat, u_cont


