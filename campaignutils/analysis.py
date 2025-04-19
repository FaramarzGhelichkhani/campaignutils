import pandas as pd 
from typing import Union, List, Dict
from ambdutils.portfolio import PortfolioProvider
from campaignutils.customer_similarity import SimilarFinder
from campaignutils.statistical import  StatisticalTests
pd.set_option('future.no_silent_downcasting', True)

class CampaignAnalytics:
    def __init__(self,
                 name: str, 
                 target_customers: List , 
                 control_customers: List=[],
                 before_campaign_date_keys: List[int]=[14030602],
                 after_campaign_date_keys: List[int]=[14030802],
                 issuer_keys : Union[bool, Dict[str, List[int]]] = True,
                 fund_keys: Union[bool, Dict[str, List[int]]] = True,
                 customer_batch_size:int = 15000, 
                 end_customerkey:int = 20200000,
                 customer_offset:int = 100000,
                 processor_number:int = 10
                   ):
        self.name = name
        self.target_customerkeys = target_customers
        self.control_customerkeys= control_customers
        self.issuer_keys = issuer_keys
        self.fund_keys = fund_keys
        self.before_campaign_date_keys =before_campaign_date_keys
        self.after_campaign_date_keys  =after_campaign_date_keys
        self.before_data_all=None
        self.end_customerkey = end_customerkey
        self.batch_size = customer_batch_size
        self.step = customer_offset 
        self.processor_number = processor_number
        self.dfs={}

        if len(self.target_customerkeys) == 0:
            raise ValueError("target is not specified")

    def fetch_data(self, group:str ="target"):
        """ 
        collect data for customers regarding features and datekeys
        :param grpup: target or control
        :return:  data  
        """
        customers = []
        if group == "target":
            customers = self.target_customerkeys
        elif group == "control":
            if len(self.control_customerkeys) == 0:
                raise ValueError("control is not specified")
            customers = self.control_customerkeys
        else:
            raise ValueError("wrong value forg group. target or control is correct value.")

 
        before_df = self._get_portfo_all_date(date_list=self.before_campaign_date_keys, customers=customers, date_type="before", group=group)
        after_df  = self._get_portfo_all_date(date_list=self.after_campaign_date_keys,  customers=customers, date_type="after", group=group)
        

        self.dfs[group] = pd.concat([before_df, after_df], axis=0)

    def _get_portfo_single_date(self, date, customers, group="target", date_type="before"):
        """
        customer_type can be target or control
        date_type can be before or after
        """
        obj = PortfolioProvider(datekey=date, issuer_keys=self.issuer_keys, fund_keys=self.fund_keys, customer_codes =customers, customer_batch_size=self.batch_size, customer_offset=self.step, end_customerkey=self.end_customerkey, processor_number=self.processor_number)
        df = obj.get_portfolios()
        df = df.assign(datekey= date, group= group, date_type=date_type)
        return df
    
    def _get_portfo_all_date(self, date_list, customers, date_type , group):
        dfs = []
        for datekey in date_list:
                df = self._get_portfo_single_date(date=datekey, customers=customers, group=group, date_type=date_type)
                dfs.append(df)
        return pd.concat(dfs, axis=0)
        
    def find_control_customers(self, numeric_features, categorical_features, n_neighbors=1, key_feature="customerCode"):
        """ 
        find similar customer regarding target customers.
        this function use customer_similarity module and target_data.
        set control_customerkeys attribute. 
        :return control_data  
        """
        if not isinstance(self.before_data_all, pd.DataFrame):
            self.before_data_all = self._get_portfo_all_date(date_list=self.before_campaign_date_keys, customers = None , date_type="before", group="no grouped")
          
        drop_index = self.before_data_all[self.before_data_all.customerCode.isin(self.target_customerkeys)].index
        data = self.before_data_all.drop(drop_index, axis=0)
        similar_customers = SimilarFinder.find_neighbors(target_data= data.fillna(0), \
                                    sample_data=self.dfs["target"][self.dfs["target"].date_type =="before"].fillna(0),\
                                    numeric_features=numeric_features,\
                                    categorical_features= categorical_features ,\
                                    n_neighbors= n_neighbors,\
                                    key_feature=key_feature)
        
        self.control_customerkeys = list(set(similar_customers))
        return self.before_data_all[self.before_data_all.customerCode.isin(list(set(similar_customers)))]    
    
    def statistical_test(self,feature:str):
        """
        use statistical module and apply on for target and control data.
        :param: feature: feature name
        :param type: can be independent and paired
        :return: result of statistical test, t, p_value 
        """
        target_data = self._get_diff_df(feature=feature, group="target")
        control_data = self._get_diff_df(feature=feature, group="control")

        # targetr part
        print(f'target instancce: {target_data.shape[0]}')
        t, pval =StatisticalTests.t_test_rel(data1=target_data[f'{feature}_after'].values, data2=target_data[f'{feature}_before'].values, alternative="greater") 
        print(f'paired t test for target data:\nt: {t}, p-value:{pval}')
        # control part 
        print(f'control instancce: {control_data.shape[0]}')
        t, pval =StatisticalTests.t_test_rel(data1=control_data[f'{feature}_after'].values, data2=control_data[f'{feature}_before'].values, alternative="greater") 
        print(f'paired t test for control_data data:\nt: {t}, p-value:{pval}')
        # diff part
        t, pval = StatisticalTests.t_test_ind(data1=target_data, data2=control_data, feature=f"{feature}_diff", alternative="greater") 
        print(f'independent t test for diff value of {feature} between target and control:\nt: {t}, p-value:{pval}')

        return t, pval

    def _get_diff_df(self, feature,  group):
        
        customers = []
        if group == "target":
            customers = self.target_customerkeys
        elif group == "control":
            if len(self.control_customerkeys) == 0:
                raise ValueError("control is not specified")
            customers = self.control_customerkeys
        else:
            raise ValueError("wrong value forg group. target or control is correct value.")
        
        init_df  = self.dfs[group].copy()
        init_df['date_type'] = pd.Categorical(init_df['date_type'], categories=['after', 'before'], ordered=True)
        init_df = init_df.sort_values(by=['customerCode', 'date_type'])

        result = pd.DataFrame({'customerCode': customers})
        customer_age = init_df.groupby(['customerCode'])['age'].max().reset_index() 
        pivoted_df = pd.pivot_table(init_df,values=feature, index=['customerCode', 'genderKey'], columns='date_type', aggfunc='mean' ).fillna(0)
        pivoted_df.columns = [f'{feature}_after', f'{feature}_before']
        pivoted_df.reset_index(inplace=True)
        result = result.merge(pivoted_df, how='left', left_on='customerCode', right_on='customerCode')
        result = result.merge(customer_age, on="customerCode", how="left")
        result[f'{feature}_after'] = result[f'{feature}_after'].fillna(0)
        result[f'{feature}_before'] = result[f'{feature}_before'].fillna(0)
        result[f'{feature}_diff'] = result[f'{feature}_after'] - result[f'{feature}_before']
        return result 
        
    def _get_diff_df_multi_features(self, features, group):
        out=[]
        for feature in features:
            df = self._get_diff_df(feature=feature, group=group)
            out.append(df)
        
        merged_df = out[0]
        for df in out[1:]:
            merged_df = pd.merge(merged_df, df, on=('customerCode',"genderKey", "age"), how="inner")
        return merged_df
    
    def to_excel(self, features:List[str]):
        """
        make excel file of target and control file
        param: features: list of feture names
        """
        target_df = self._get_diff_df_multi_features(features=features, group="target")
        control_df = self._get_diff_df_multi_features(features=features, group="control")
        file_name = f"{self.name}_data.xlsx"
        sheet_name_1 = 'target'
        sheet_name_2 = 'control'
        with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
            target_df.to_excel(writer, sheet_name=sheet_name_1, index=False)
            control_df.to_excel(writer, sheet_name=sheet_name_2, index=False)

    # def algorithm(self):
    #     # fetch_data(target)
    #     # find_control_customers
    #     # fetch_data(control)
    #     # statistical result
    #     # to excel     
    #     pass      
