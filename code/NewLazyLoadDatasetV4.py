import itertools
import os
import copy

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Dataset, HeteroData, Data
import torch_geometric.transforms as T
from sklearn.utils import resample


generic_callgraph_edge = "CALL"
generic_other_edges = "OTHER"
generic_pdg_edge = "PDG"
generic_ast_edge = "AST"
generic_cfg_edge = "CFG"
generic_self_edge = "SELF"

edge_type_mapping = {
    "AST": "AST",
    "CONDITION": "AST",
    "REF": "AST",
    "CFG": "CFG",
    "CDG": "PDG",
    "REACHING_DEF": "PDG",
    "CALL": "CALL",
    "ARGUMENT": "CALL",
    "RECEIVER": "CALL"
}

def get_edge_type(edge_type):
    et = edge_type_mapping[edge_type] if edge_type in edge_type_mapping else "OTHER"
    return et

ast_features = ['current_AST_compilationunit', 'current_AST_switchstatementcase', 'current_AST_enhancedforcontrol', 'current_AST_dostatement', 'delta_AST_localvariabledeclaration', 'current_AST_localvariabledeclaration', 'delta_AST_explicitconstructorinvocation', 'current_AST_superconstructorinvocation', 'current_AST_constructordeclaration', 'delta_AST_fielddeclaration', 'current_AST_supermemberreference', 'delta_AST_elementvaluepair', 'current_AST_typeargument', 'current_AST_tryresource', 'delta_AST_packagedeclaration',  'current_AST_blockstatement', 'delta_AST_trystatement', 'current_AST_methodreference', 'current_AST_primary', 'delta_AST_memberreference', 'current_AST_classreference', 'delta_AST_classdeclaration', 'delta_AST_enumconstantdeclaration', 'current_AST_statementexpression', 'delta_AST_lambdaexpression', 'current_AST_type', 'delta_AST_annotationdeclaration', 'delta_AST_forstatement', 'current_AST_continuestatement', 'current_AST_forstatement', 'current_AST_methodinvocation', 'parent_AST_memberreference', 'parent_AST_variabledeclarator', 'parent_AST_formalparameter', 'delta_AST_continuestatement', 'current_AST_this', 'delta_AST_enumbody', 'current_AST_synchronizedstatement', 'delta_AST_supermethodinvocation', 'parent_AST_breakstatement', 'current_AST_methoddeclaration', 'delta_AST_literal', 'delta_AST_primary', 'current_AST_annotationdeclaration']
ast_features = ['current_AST_compilationunit', 'delta_AST_referencetype', 'delta_AST_statementexpression', 'current_AST_switchstatementcase', 'current_AST_enhancedforcontrol', 'current_AST_dostatement', 'delta_AST_localvariabledeclaration', 'current_AST_localvariabledeclaration', 'delta_AST_explicitconstructorinvocation', 'current_AST_superconstructorinvocation', 'current_AST_constructordeclaration', 'delta_AST_fielddeclaration', 'current_AST_supermemberreference', 'delta_AST_elementvaluepair', 'current_AST_typeargument', 'current_AST_tryresource', 'delta_AST_packagedeclaration', 'parent_AST_interfacedeclaration', 'current_AST_blockstatement', 'delta_AST_trystatement', 'current_AST_methodreference', 'current_AST_primary', 'delta_AST_memberreference', 'current_AST_classreference', 'delta_AST_classdeclaration', 'delta_AST_enumconstantdeclaration', 'current_AST_statementexpression', 'delta_AST_lambdaexpression', 'delta_AST_variabledeclaration', 'current_AST_type', 'delta_AST_annotationdeclaration', 'delta_AST_forstatement', 'delta_AST_tryresource', 'current_AST_continuestatement', 'current_AST_forstatement', 'current_AST_methodinvocation', 'parent_AST_memberreference', 'parent_AST_variabledeclarator', 'parent_AST_annotationdeclaration', 'parent_AST_formalparameter', 'delta_AST_continuestatement', 'current_AST_this', 'delta_AST_enumbody', 'current_AST_synchronizedstatement', 'delta_AST_supermethodinvocation', 'parent_AST_breakstatement', 'current_AST_methoddeclaration', 'delta_AST_literal', 'delta_AST_primary', 'current_AST_annotationdeclaration']

pmd_features = ['current_PMD_rule_type_strict exception rules', 'current_PMD_severity_minor', 'delta_PMD_vnc', 'current_PMD_arp', 'current_PMD_severity_major', 'parent_PMD_severity_major', 'delta_PMD_severity_major', 'current_PMD_severity_critical', 'parent_PMD_severity_critical', 'delta_PMD_severity_critical', 'parent_PMD_severity_minor', 'current_PMD_dp', 'delta_PMD_severity_minor', 'current_PMD_rule_type_basic rules', 'parent_PMD_rule_type_basic rules', 'delta_PMD_rule_type_basic rules', 'parent_PMD_arp', 'parent_PMD_rule_type_strict exception rules', 'delta_PMD_rule_type_strict exception rules', 'current_PMD_rule_type_string and stringbuffer rules', 'parent_PMD_dp', 'parent_PMD_vnc', 'parent_PMD_fdsbasoc', 'current_PMD_apmp', 'current_PMD_gdl', 'current_PMD_aio', 'current_PMD_scn', 'parent_PMD_aio', 'parent_PMD_scn', 'delta_PMD_scn', 'current_PMD_ismub', 'parent_PMD_adl', 'current_PMD_adl', 'current_PMD_vnc', 'parent_PMD_uv', 'current_PMD_uv', 'current_PMD_loc', 'current_PMD_atret', 'delta_PMD_if', 'current_PMD_atnpe', 'parent_PMD_if', 'current_PMD_if', 'parent_PMD_rule_type_string and stringbuffer rules', 'current_PMD_gls', 'parent_PMD_rule_type_javabean rules', 'delta_PMD_rule_type_type resolution rules', 'current_PMD_rule_type_controversial rules', 'parent_PMD_rule_type_controversial rules', 'delta_PMD_rule_type_controversial rules', 'current_PMD_rule_type_jakarta commons logging rules', 'parent_PMD_rule_type_jakarta commons logging rules', 'delta_PMD_rule_type_jakarta commons logging rules', 'current_PMD_rule_type_type resolution rules', 'parent_PMD_rule_type_type resolution rules', 'delta_PMD_uec', 'current_PMD_rule_type_naming rules', 'parent_PMD_rule_type_clone implementation rules', 'parent_PMD_uec', 'current_PMD_uec', 'current_PMD_acge', 'current_PMD_rule_type_java logging rules', 'parent_PMD_rule_type_java logging rules', 'parent_PMD_rfi', 'delta_PMD_rfi', 'parent_PMD_rule_type_naming rules', 'delta_PMD_rule_type_naming rules', 'parent_PMD_rule_type_brace rules', 'current_PMD_rule_type_optimization rules', 'parent_PMD_rule_type_design rules', 'current_PMD_rule_type_design rules', 'current_PMD_rfi', 'parent_PMD_rule_type_unnecessary and unused code rules', 'current_PMD_rule_type_brace rules', 'delta_PMD_rule_type_design rules', 'parent_PMD_rule_type_optimization rules', 'parent_PMD_rule_type_security code guideline rules', 'delta_PMD_rule_type_optimization rules', 'parent_PMD_acge', 'current_PMD_asaml', 'current_PMD_fdsbasoc', 'parent_PMD_rule_type_junit rules', 'parent_PMD_rule_type_j2ee rules', 'parent_PMD_rule_type_finalizer rules', 'parent_PMD_gls', 'parent_PMD_apmp', 'delta_PMD_loc', 'current_PMD_uni', 'delta_PMD_dp', 'delta_PMD_uni', 'parent_PMD_gdl', 'parent_PMD_rule_type_import statement rules', 'delta_PMD_fdsbasoc', 'parent_PMD_asaml', 'current_PMD_uem', 'parent_PMD_atret', 'current_PMD_rule_type_unnecessary and unused code rules', 'current_PMD_pst', 'current_PMD_ucie', 'delta_PMD_rule_type_unnecessary and unused code rules', 'delta_PMD_rule_type_string and stringbuffer rules']
pmd_features = ['current_PMD_rule_type_strict exception rules', 'current_PMD_severity_minor', 'delta_PMD_vnc', 'current_PMD_arp', 'current_PMD_severity_major', 'parent_PMD_severity_major', 'delta_PMD_severity_major', 'current_PMD_severity_critical', 'parent_PMD_severity_critical', 'delta_PMD_severity_critical', 'parent_PMD_severity_minor', 'current_PMD_dp', 'delta_PMD_severity_minor', 'current_PMD_rule_type_basic rules', 'parent_PMD_rule_type_basic rules', 'delta_PMD_rule_type_basic rules', 'parent_PMD_arp', 'parent_PMD_rule_type_strict exception rules', 'delta_PMD_rule_type_strict exception rules', 'current_PMD_rule_type_string and stringbuffer rules', 'parent_PMD_dp', 'parent_PMD_vnc', 'parent_PMD_fdsbasoc', 'current_PMD_apmp', 'current_PMD_gdl', 'current_PMD_aio', 'current_PMD_scn', 'parent_PMD_aio', 'parent_PMD_scn', 'delta_PMD_scn', 'current_PMD_ismub', 'parent_PMD_adl', 'current_PMD_adl', 'current_PMD_vnc', 'parent_PMD_uv', 'current_PMD_uv', 'current_PMD_loc', 'current_PMD_atret', 'delta_PMD_if', 'current_PMD_atnpe', 'parent_PMD_if', 'current_PMD_if', 'parent_PMD_rule_type_string and stringbuffer rules', 'current_PMD_gls', 'parent_PMD_rule_type_javabean rules', 'delta_PMD_rule_type_type resolution rules', 'current_PMD_rule_type_controversial rules', 'parent_PMD_rule_type_controversial rules', 'delta_PMD_rule_type_controversial rules', 'current_PMD_rule_type_jakarta commons logging rules', 'parent_PMD_rule_type_jakarta commons logging rules', 'delta_PMD_rule_type_jakarta commons logging rules', 'current_PMD_rule_type_type resolution rules', 'parent_PMD_rule_type_type resolution rules', 'delta_PMD_uec', 'current_PMD_rule_type_naming rules', 'parent_PMD_rule_type_clone implementation rules', 'parent_PMD_uec', 'current_PMD_uec', 'current_PMD_acge', 'current_PMD_rule_type_java logging rules', 'parent_PMD_rule_type_java logging rules', 'parent_PMD_rfi', 'delta_PMD_rfi', 'parent_PMD_rule_type_naming rules', 'delta_PMD_rule_type_naming rules', 'parent_PMD_rule_type_brace rules', 'current_PMD_rule_type_optimization rules', 'parent_PMD_rule_type_design rules', 'current_PMD_rule_type_design rules', 'current_PMD_rfi', 'parent_PMD_rule_type_unnecessary and unused code rules', 'current_PMD_rule_type_brace rules', 'delta_PMD_rule_type_design rules', 'parent_PMD_rule_type_optimization rules', 'parent_PMD_rule_type_security code guideline rules', 'delta_PMD_rule_type_optimization rules', 'parent_PMD_acge', 'current_PMD_asaml', 'current_PMD_fdsbasoc', 'parent_PMD_rule_type_junit rules', 'parent_PMD_rule_type_j2ee rules', 'parent_PMD_rule_type_finalizer rules', 'parent_PMD_gls', 'parent_PMD_apmp', 'delta_PMD_loc', 'current_PMD_uni', 'delta_PMD_dp', 'delta_PMD_uni', 'parent_PMD_gdl', 'parent_PMD_rule_type_import statement rules', 'delta_PMD_fdsbasoc', 'parent_PMD_asaml', 'current_PMD_uem', 'parent_PMD_atret', 'current_PMD_rule_type_unnecessary and unused code rules', 'current_PMD_pst', 'current_PMD_ucie', 'delta_PMD_rule_type_unnecessary and unused code rules', 'delta_PMD_rule_type_string and stringbuffer rules']

sm_features = ['current_SM_interface_ng_sum', 'current_SM_method_mism_median', 'current_SM_class_loc_stdev', 'current_SM_method_mi_avg', 'delta_SM_enum_tng_theil', 'current_SM_method_mi_max', 'current_SM_interface_tnlm_sum', 'current_SM_class_tnos_stdev', 'current_SM_package_tnm', 'current_SM_method_misei_avg', 'delta_SM_class_tnls_stdev', 'current_SM_enum_tng_generalized_entropy', 'current_SM_method_misei_median', 'current_SM_interface_dloc_median', 'delta_SM_enum_tng_generalized_entropy', 'current_SM_file_loc', 'current_SM_file_pda', 'current_SM_class_tcloc_avg', 'current_SM_class_tloc_stdev', 'current_SM_method_loc_median', 'current_SM_class_tcloc_median', 'current_SM_method_loc_max', 'current_SM_method_mi_median', 'parent_SM_class_tloc_stdev', 'current_SM_class_lloc_median', 'current_SM_class_lloc_min', 'delta_SM_interface_tng_min', 'delta_SM_class_ng_coefficient_of_variation', 'current_SM_class_ccl_min', 'current_SM_method_misei_min', 'delta_SM_method_tlloc_avg', 'current_SM_class_nlpm_min', 'current_SM_file_cloc', 'current_SM_class_na_max', 'current_SM_class_cloc_median', 'current_SM_class_lloc_avg', 'delta_SM_class_na_max', 'current_SM_method_tnos_generalized_entropy', 'current_SM_class_tcloc_max', 'current_SM_interface_tnm_median', 'current_SM_class_na_sum', 'delta_SM_method_mccc_min', 'current_SM_method_mism_generalized_entropy', 'current_SM_method_mims_max', 'current_SM_interface_ng_max', 'current_SM_class_nlpm_sum', 'current_SM_method_nle_avg', 'current_SM_interface_nm_median', 'delta_SM_class_lloc_coefficient_of_variation', 'parent_SM_class_nos_median', 'current_SM_method_nii_sum', 'current_SM_method_nos_max', 'current_SM_class_ng_generalized_entropy', 'current_SM_method_mims_median', 'current_SM_class_ldc_min', 'delta_SM_method_misei_max', 'delta_SM_class_pda_median', 'current_SM_class_dloc_max', 'current_SM_method_hpl_generalized_entropy', 'delta_SM_class_nla_max', 'current_SM_class_nos_median', 'current_SM_interface_tnlm_max', 'current_SM_class_dloc_generalized_entropy', 'current_SM_class_ldc_sum', 'delta_SM_interface_ng_sum', 'current_SM_class_na_min', 'current_SM_annotation_tnla_generalized_entropy', 'current_SM_class_nlpm_median', 'current_SM_class_tlloc_min', 'current_SM_class_tna_theil', 'current_SM_class_cd_avg', 'current_SM_class_nle_sum', 'current_SM_enum_tnlm_avg', 'delta_SM_interface_tnlm_sum', 'delta_SM_method_cd_generalized_entropy', 'current_SM_interface_npm_median', 'current_SM_class_tnpa_avg', 'current_SM_class_tng_median', 'current_SM_class_nle_max', 'delta_SM_method_tloc_sum', 'current_SM_class_ng_stdev', 'parent_SM_class_cloc_min', 'delta_SM_class_ng_stdev', 'current_SM_interface_loc_stdev', 'current_SM_interface_tloc_stdev', 'current_SM_annotation_ldc_median', 'current_SM_interface_cloc_stdev', 'delta_SM_class_nla_min', 'current_SM_interface_tng_avg', 'current_SM_interface_tnlm_median', 'current_SM_interface_nm_min', 'current_SM_annotation_tnla_shannon_entropy', 'current_SM_class_nm_theil', 'delta_SM_interface_dit_median', 'current_SM_interface_tnpm_median', 'delta_SM_interface_tnpm_median', 'parent_SM_class_nos_stdev', 'delta_SM_interface_tng_stdev', 'delta_SM_annotation_cboi_shannon_entropy', 'current_SM_enum_nls_sum']
sm_features = ['current_SM_interface_ng_sum', 'current_SM_method_mism_median', 'current_SM_class_loc_stdev', 'current_SM_method_mi_avg', 'delta_SM_enum_tng_theil', 'current_SM_method_mi_max', 'current_SM_interface_tnlm_sum', 'current_SM_class_tnos_stdev', 'current_SM_package_tnm', 'current_SM_method_misei_avg', 'delta_SM_class_tnls_stdev', 'current_SM_enum_tng_generalized_entropy', 'current_SM_method_misei_median', 'current_SM_interface_dloc_median', 'delta_SM_enum_tng_generalized_entropy', 'current_SM_file_loc', 'current_SM_file_pda', 'current_SM_class_tcloc_avg', 'current_SM_class_tloc_stdev', 'current_SM_method_loc_median', 'current_SM_class_tcloc_median', 'current_SM_method_loc_max', 'current_SM_method_mi_median', 'parent_SM_class_tloc_stdev', 'current_SM_class_lloc_median', 'current_SM_class_lloc_min', 'delta_SM_interface_tng_min', 'delta_SM_class_ng_coefficient_of_variation', 'current_SM_class_ccl_min', 'current_SM_method_misei_min', 'delta_SM_method_tlloc_avg', 'current_SM_class_nlpm_min', 'current_SM_file_cloc', 'current_SM_class_na_max', 'current_SM_class_cloc_median', 'current_SM_class_lloc_avg', 'delta_SM_class_na_max', 'current_SM_method_tnos_generalized_entropy', 'current_SM_class_tcloc_max', 'current_SM_interface_tnm_median', 'current_SM_class_na_sum', 'delta_SM_method_mccc_min', 'current_SM_method_mism_generalized_entropy', 'current_SM_method_mims_max', 'current_SM_interface_ng_max', 'current_SM_class_nlpm_sum', 'current_SM_method_nle_avg', 'current_SM_interface_nm_median', 'delta_SM_class_lloc_coefficient_of_variation', 'parent_SM_class_nos_median', 'current_SM_method_nii_sum', 'current_SM_method_nos_max', 'current_SM_class_ng_generalized_entropy', 'current_SM_method_mims_median', 'current_SM_class_ldc_min', 'delta_SM_method_misei_max', 'delta_SM_class_pda_median', 'current_SM_class_dloc_max', 'current_SM_method_hpl_generalized_entropy', 'delta_SM_class_nla_max', 'current_SM_class_nos_median', 'current_SM_interface_tnlm_max', 'current_SM_class_dloc_generalized_entropy', 'current_SM_class_ldc_sum', 'delta_SM_interface_ng_sum', 'current_SM_class_na_min', 'current_SM_annotation_tnla_generalized_entropy', 'current_SM_class_nlpm_median', 'current_SM_class_tlloc_min', 'current_SM_class_tna_theil', 'current_SM_class_cd_avg', 'current_SM_class_nle_sum', 'current_SM_enum_tnlm_avg', 'delta_SM_interface_tnlm_sum', 'delta_SM_method_cd_generalized_entropy', 'current_SM_interface_npm_median', 'current_SM_class_tnpa_avg', 'current_SM_class_tng_median', 'current_SM_class_nle_max', 'delta_SM_method_tloc_sum', 'current_SM_class_ng_stdev', 'parent_SM_class_cloc_min', 'delta_SM_class_ng_stdev', 'current_SM_interface_loc_stdev', 'current_SM_interface_tloc_stdev', 'current_SM_annotation_ldc_median', 'current_SM_interface_cloc_stdev', 'delta_SM_class_nla_min', 'current_SM_interface_tng_avg', 'current_SM_interface_tnlm_median', 'current_SM_interface_nm_min', 'current_SM_annotation_tnla_shannon_entropy', 'current_SM_class_nm_theil', 'delta_SM_interface_dit_median', 'current_SM_interface_tnpm_median', 'delta_SM_interface_tnpm_median', 'parent_SM_class_nos_stdev', 'delta_SM_interface_tng_stdev', 'delta_SM_annotation_cboi_shannon_entropy', 'current_SM_enum_nls_sum']

kamei = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev', 'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']

all_manual_features = kamei + ast_features + sm_features + pmd_features
# all_manual_features = kamei
all_manual_features_size = len(all_manual_features)

generic_callgraph_edge = "CALL"
generic_other_edges = "OTHER"
generic_pdg_edge = "PDG"
generic_ast_edge = "AST"
generic_cfg_edge = "CFG"
generic_self_edge = "SELF"

def empty_data():
    data = Data()
    data.x = torch.empty((0, 769 + 128))  # Replace `feature_dim` with the dimensionality of your features
    data.edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
    data.node_type = torch.empty((0,),
                                 dtype=torch.long)  # Use long if node types are represented as integers
    data.edge_type = torch.empty((0,), dtype=torch.long)
    data.id = torch.empty((0,),dtype=torch.long)
    return data
    
def to_homogeneous(graph_data):
    if graph_data.num_nodes == 0 or graph_data.num_edges == 0 or len(graph_data.edge_types) == 0:
        data = empty_data()
        return data
    homogeneous_data = graph_data.to_homogeneous(
        node_attrs=['x','id'],
        add_node_type=True,
        add_edge_type=True,
        dummy_values=True
    )
    return homogeneous_data

def transformation(sample):
    graph, commit_message, code, feature_embedding, label, commit = sample
    graph = T.NormalizeFeatures()(graph)
    graph = to_homogeneous(graph)
    return graph, commit_message, code,feature_embedding, label, commit


def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df

class NewLazyLoadDatasetV4(Dataset):

    def __init__(self, data_dir, projects=None, data_type='train', merge=True, old=True,
                 tokenizer=None, model=None, device='cpu', changes_data=None, features_data=None, need_attentions=True, under_sampling=False, scaler=None):
        """
        Args:
            data_dir (string): Directory with all the graphs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.msg_length = 64
        self.scaler = StandardScaler() if scaler is None else scaler
        self.all_manual_features = all_manual_features
        self.all_manual_features_size = len(all_manual_features)
        self.data_dir = data_dir
        self.projects = projects
        self.data_type = data_type
        self.cached_data = {}
        self.device = device
        self.changes_data = changes_data
        self.features_data = features_data
        self.need_attentions = need_attentions
        self.under_sampling = under_sampling
        self.text_attns = {}
        if old:
            self.added_path_suffix = '_added_graph_p_1000_hetero_data.pt'
            self.removed_path_suffix = '_removed_graph_p_1000_hetero_data.pt'
        else:
            self.added_path_suffix = '_after_graphml_hetero_data_change_only.pt'
            self.removed_path_suffix = '_before_graphml_hetero_data_change_only.pt'
            self.path_suffix = '_hetero_data_1.pt'
        self.merge = merge
        self.bert_tokenizer = tokenizer
        self.bert_model = model
        if projects is not None:
            all_labels = []
            for project in projects:
                data = pd.read_csv(data_dir + '/' + data_type + '/' + project + '_' + data_type + '_enhanced.csv', low_memory=False)
                data = data.sort_values(by='author_date')
                data['code'] = data['commit_hash'].apply(lambda x: self.changes_data.loc[x][2])
                data['commit_hash'] = project + '_' + data['commit_hash']
                all_labels.append(data)
            self.labels = pd.concat(all_labels)

        features = ['commit_hash', 'is_buggy_commit', 'commit_message', 'code']
        features += self.all_manual_features
        self.labels = self.labels[features]
        man_features = self.labels[self.all_manual_features]
        string_columns = man_features.select_dtypes(include=['object']).columns
        for col in string_columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        man_features = man_features.fillna(0)
        man_features = convert_dtype_dataframe(man_features, self.all_manual_features)
        # if self.data_type == 'train':
        #     man_features[self.all_manual_features] = self.scaler.fit_transform(man_features[self.all_manual_features].to_numpy())
        # else:
        #     man_features[self.all_manual_features] = self.scaler.transform(man_features[self.all_manual_features].to_numpy())
        man_features[self.all_manual_features] = preprocessing.scale(man_features[self.all_manual_features].to_numpy())
        self.labels[self.all_manual_features] = man_features
        self.labels['commit_message'] = self.labels['commit_message'].fillna('commit message').astype(str)
        self.labels['commit_message'] = self.labels['commit_message'].apply(lambda x: str(x))

        # shuffle labels
        if data_type == 'train':
            self.labels = self.labels.sample(frac=1)
        self.labels.reset_index(inplace=True)
        self.labels.set_index('commit_hash', inplace=True)
        self.verify_data()
        self.labels_bak = copy.deepcopy(self.labels)
        self.pos_count = self.labels['is_buggy_commit'].sum()
        self.neg_count = len(self.labels) - self.labels['is_buggy_commit'].sum()
        print(f"Data type: {self.data_type}, pos count: {self.pos_count}, neg count: {self.neg_count}")

    def do_under_sampling(self):
        if self.data_type == 'train':
            self.labels = copy.deepcopy(self.labels_bak)
            self.labels.reset_index(inplace=True)
            pos_samples = self.labels[self.labels['is_buggy_commit'] > 0].reset_index(drop=True)
            neg_samples = self.labels[self.labels['is_buggy_commit'] < 1].reset_index(drop=True)

            neg_samples_resampled = resample(neg_samples, 
                                         n_samples=len(pos_samples),  # Resample to match the size of positive samples
                                         replace=False)  # No replacement, purely random undersampling
        
            # Step 3: Combine the resampled negative samples with the positive samples
            balanced_data = pd.concat([pos_samples, neg_samples_resampled])
            
            # Step 4: Reset the index to ensure a clean DataFrame
            self.labels = balanced_data.reset_index(drop=True)
            self.labels.set_index('commit_hash', inplace=True)
            
    def text_embeddings(self, commit_msg, code):
        # source
        added_tokens = []
        removed_tokens = []
        msg_tokens = self.bert_tokenizer.tokenize(commit_msg)
        msg_tokens = msg_tokens[:min(self.msg_length, len(msg_tokens))]
        # for file_codes in files:
        file_codes = code
        added_codes = [' '.join(line.split()) for line in file_codes['added_code']]
        removed_codes = [' '.join(line.split()) for line in file_codes['removed_code']]

        codes = '[ADD]'.join([line for line in added_codes if len(line)])
        added_tokens.extend(self.bert_tokenizer.tokenize(codes))

        codes = '[DEL]'.join([line for line in removed_codes if len(line)])
        removed_tokens.extend(self.bert_tokenizer.tokenize(codes))

        input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens

        input_tokens = input_tokens[:512 - 2]

        tokens = [self.bert_tokenizer.cls_token] + input_tokens + [self.bert_tokenizer.sep_token]
        tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(tokens_ids)

        # Convert to tensors and add batch dimension (batch size = 1)
        input_ids = torch.tensor(tokens_ids).unsqueeze(0)  # shape: (1, sequence_length)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # shape: (1, sequence_length)

        # Ensure tensors are on the correct device (CPU or GPU)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Perform inference with no gradient tracking (for efficiency)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the hidden state of the [CLS] token (index 0) from the last layer
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        last_layer_attentions = outputs.attentions[-1]

        return {
            'cls_embedding': cls_embedding.squeeze(0).cpu().numpy(),
            'attn_weights': last_layer_attentions.cpu().numpy(),
            'tokens': tokens
        }

    def verify_data(self):
        print(f"verifying data {self.data_type}")
        for commit_hash in self.labels.index:
            try:
                project_name = commit_hash.split('_')[0]
                commit_id = commit_hash.split('_')[1]
                graph_path = os.path.join(self.data_dir, project_name,
                                                commit_id + self.path_suffix)

                if not os.path.exists(graph_path):
                    print(f"drop missing {graph_path}", flush=True)
                    self.labels.drop(commit_hash, inplace=True)
                # else:
                #     count_non_empty = 0
                #     try:
                #         if os.path.exists(graph_path):
                #             # print(f" added_graph_path {added_graph_path}", flush=True)
                #             graph = torch.load(graph_path, weights_only=False)
                #             if graph.num_nodes > 0:
                #                 count_non_empty += 1
                #     except Exception as e:
                #         print(f"error reading file {graph_path}, drop the commit",
                #               flush=True)
                #         self.labels.drop(commit_hash, inplace=True)
                #         count_non_empty = -1
                #     if (self.data_type == 'train' or self.labels.at[
                #         commit_hash, 'is_buggy_commit'] == 1) and count_non_empty == 0:
                #         print(f"Dropping wrong data: buggy commit {commit_hash} due to no nodes in graphs", flush=True)
                #         self.labels.drop(commit_hash, inplace=True)

            except Exception as e:
                print(f"{commit_hash} {e}", flush=True)

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels['is_buggy_commit'].tolist()

    def get_projects(self):
        try:
            return [commit_hash.split('_')[0] for commit_hash in self.labels.index]
        except Exception as e:
            print(self)
    
    def set_config(self, manual_features_columns, graph_edges):
        self.graph_edges = graph_edges
        self.manual_features_columns = manual_features_columns
        features = ['is_buggy_commit', 'commit_message', 'code']
        features += self.manual_features_columns
        self.labels = self.labels[features]

    def prepare_data(self, data):
        if (self.manual_features_columns is None) or (self.graph_edges is None):
            print("no values manual_features_columns and graph_edges")
            return data
        manual_features_columns = self.manual_features_columns
        graph_edges = self.graph_edges
        
        graph, commit_message, code, feature_embedding, label, commit_hash = data
        graph_edges_tr = '_'.join(graph_edges) if graph_edges is not None else ''
        project_name = commit_hash.split('_')[0]
        commit_id = commit_hash.split('_')[1]
        if graph_edges is not None:
            edge_types = [et for et in graph.edge_types if get_edge_type(et[1]) in self.graph_edges]
            graph = graph.edge_type_subgraph(edge_types=edge_types)

        sample = (graph, commit_message, code, feature_embedding, label, commit_hash)
        num_to_node_id_map = {}
        node_id_to_num_map = {}

        i = 0
        for node_storage in graph.node_stores:
            ids = []
            for node_id in node_storage.id:
                num_to_node_id_map[i] = node_id
                node_id_to_num_map[node_id] = id
                ids.append(i)
                i += 1
            node_storage.id = torch.tensor(ids)

        id_mapping_path = os.path.join(self.data_dir, project_name, commit_id + graph_edges_tr + '_id_mapping.pt')
        torch.save((num_to_node_id_map, node_id_to_num_map), id_mapping_path)

        sample = transformation(sample)
        return sample
        
    def get_item_from_commit_hash(self, commit_hash):
        label = self.labels.at[commit_hash, 'is_buggy_commit']
        commit_message = str(self.labels.at[commit_hash, 'commit_message'])
        code = str(self.labels.at[commit_hash, 'code'])
        features_columns = self.manual_features_columns if self.manual_features_columns is not None else self.all_manual_features

        graph = empty_data()
        if commit_hash in self.cached_data:
            graph, feature_embedding  =  self.cached_data[commit_hash]
        else:
            # print(f"commit_hash {commit_hash}", flush=True)
            project_name = commit_hash.split('_')[0]
            commit_id = commit_hash.split('_')[1]
            feature_embedding = torch.tensor(self.labels.loc[commit_hash][features_columns].to_numpy(dtype=np.float32)).to(self.device)
            if len(commit_hash.split('_')) == 3:
                commit_id = commit_hash.split('_')[1] + '_' + commit_hash.split('_')[2]
            if self.graph_edges is not None:
                graph_path = os.path.join(self.data_dir, project_name, commit_id + self.path_suffix)
                try:
                    graph_hetero = torch.load(graph_path, weights_only=False)
                except Exception as e:
                    # print(f"error reading added graph {added_graph_path}", flush=True)
                    graph_hetero = HeteroData()
                    
                sample = (graph_hetero, commit_message, code, feature_embedding, label, commit_hash)
                sample = self.prepare_data(sample)
                graph, commit_message, code, feature_embedding, label, commit_hash = sample
                self.cached_data[commit_hash] = (graph, feature_embedding)
                    
        sample = (graph, commit_message, code, feature_embedding, label, commit_hash)
        
        return sample

    def get_sample_weights(self):
        pos_count = self.labels['is_buggy_commit'].sum()
        neg_count = len(self.labels) - self.labels['is_buggy_commit'].sum()
        total = pos_count + neg_count
        # give weight to each sample
        pos_weight = pos_count / total
        neg_weight = neg_count / total
        sample_weights = self.labels['is_buggy_commit'].apply(lambda x: neg_weight if x == 1 else neg_weight)
        return torch.tensor(np.array(sample_weights), dtype=torch.float32)


    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.get_item_from_commit_hash(i) for i in idx]
            return batch
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, str):
            return self.get_item_from_commit_hash(idx)
        else:
            commit_hash = self.labels.index[idx]
            return self.get_item_from_commit_hash(commit_hash)

class NewLazyLoadDatasetV4Wrapper(Dataset):
    def __init__(self, inner_dataset, manual_features_columns=None, graph_edges=None):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.manual_features_columns = manual_features_columns
        self.graph_edges = graph_edges
    def __len__(self):
        return len(self.inner_dataset)

    def get_labels(self):
        return self.inner_dataset.get_labels()

    def get_projects(self):
        return self.inner_dataset.get_projects()

    def get_sample_weights(self):
        return self.inner_dataset.get_sample_weights()

    def __getitem__(self, idx):
        return self.inner_dataset.__getitem__(idx)
