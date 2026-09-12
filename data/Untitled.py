#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas matplotlib')


# In[4]:


from datasets import load_dataset

dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")


# In[5]:


dataset


# In[6]:


dataset["from_title"][0]


# In[8]:


df = dataset["by_polishing"].to_pandas()


# In[9]:


df.columns


# In[11]:


df.dtypes


# In[12]:


df["label"] = df.apply(
    lambda row: "human" if row["original_abstract"] else "ai_generated",
    axis=1
)

# Check distribution
print(df["label"].value_counts())


# In[14]:


import pandas as pd

# تحويل القسم إلى DataFrame
df = dataset["by_polishing"].to_pandas()

# النصوص البشرية
human_rows = [{"text": row["original_abstract"], "label": "human"} 
              for _, row in df.iterrows()]

# النصوص المولدة
ai_rows = []
for _, row in df.iterrows():
    ai_rows.append({"text": row["allam_generated_abstract"], "label": "ai_generated"})
    ai_rows.append({"text": row["jais_generated_abstract"], "label": "ai_generated"})
    ai_rows.append({"text": row["llama_generated_abstract"], "label": "ai_generated"})
    ai_rows.append({"text": row["openai_generated_abstract"], "label": "ai_generated"})

# دمج النصوص البشرية والمولدة في جدول واحد
df_labels = pd.DataFrame(human_rows + ai_rows)

# فحص توزيع المتغير الهدف
print(df_labels["label"].value_counts())


# In[15]:


# عدد القيم المفقودة في كل عمود
df_labels.isnull().sum()


# In[16]:


# عدد الصفوف المكررة
df_labels.duplicated().sum()


# In[17]:


df_labels["label"].value_counts()


# In[ ]:




