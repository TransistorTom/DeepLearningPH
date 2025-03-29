import pandas as pd

def extract_messages(test_data, trained_model):
    messages_list = []
    for i in range(len(test_data)):
        trained_model(test_data[i].x, test_data[i].edge_index, save_messages=True)
        for j, record in enumerate(trained_model.message_storage):
            messages_list.append(record.copy())


    df = pd.DataFrame(messages_list)
    df[['pos_i_x', 'pos_i_y']] = pd.DataFrame(df['pos_i'].tolist(), index=df.index)
    df[['pos_j_x', 'pos_j_y']] = pd.DataFrame(df['pos_j'].tolist(), index=df.index)
    df[['message_x', 'message_y']] = pd.DataFrame(df['message'].tolist(), index=df.index)

    # Drop original array columns
    df = df.drop(columns=['pos_i', 'pos_j', 'message', 'edge'])  # optional
    df['force_y'] = df['message_y'] * df['mass_i']

    return df
