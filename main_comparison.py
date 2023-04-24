import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
import base64
import io

# Functions from your original code
def percent_formatter(x, pos):
    return f"{(x + 4) / 7:.1f}"


def normplot(data1, data2, labval1, labval2):
    mu = np.mean(data1)
    sig = np.std(data1)
    n = len(data1)
    p = [(i - 0.5) / n for i in range(1, n + 1)]
    x1 = np.sort(data1)
    y1 = norm.ppf(p)
    x2 = np.sort(data2)
    y2 = norm.ppf(p)
    ticks_perc = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    ytv = np.percentile(y1, ticks_perc)
    idxr = np.arange(1, n + 1)
    if n > 200000:
        idxr = np.concatenate((np.arange(1, 1000), np.arange(1001, 10000, 5), np.arange(10001, n, 50)))

    # Cap the maximum value of idxr to the length of x and y arrays
    idxr = np.clip(idxr, 0, min(len(x1), len(x2), len(y1), len(y2)) - 1)

    fig, ax = plt.subplots()
    ylimval = (-4, 3)

if np.max(y1) < 3.1:
    ax.scatter(x1[idxr], y1[idxr], label=labval1, s=2, linewidth=0)
    ax.scatter(x2[idxr], y2[idxr], label=labval2, s=2, linewidth=0)
    ax.set_ylim(*ylimval)
else:
    ax.scatter(x1[idxr], y1[idxr], label=labval1, s=2, linewidth=0)
    ax.scatter(x2[idxr], y2[idxr], label=labval2, s=2, linewidth=0)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which='both', axis='y', alpha=0.5)
    ax.legend(loc='lower right')
    plt.xlabel("Thickness (mm)")
    plt.ylabel("Proportion of area (%) ")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(percent_formatter))
    ax.set_ylim(-4, 3)
    plt.grid(True, which='both', axis='both', alpha=0.5)
    return fig

def linplot(df1, df2, labval1, labval2):
    fig, ax = plt.subplots()
    nd1 = len(df1)
    pr1 = (2 * np.arange(1, nd1 + 1) - 1) / 2 / nd1  # hazen plot plot pos for prob exceedance
    plt1 = plt.scatter(np.sort(df1), pr1, s=2, linewidth=0,
                       label=labval1, marker='o', color='darkorange')
    if df2 is not None:
        nd2 = len(df2)
        pr2 = (2 * np.arange(1, nd2 + 1) - 1) / 2 / nd2  # hazen plot plot pos for prob exceedance
        plt2 = plt.scatter(np.sort(df2), pr2, s=2, linewidth=0,
                           label=labval2, marker='o', color='darkblue')
    plt.xlabel("Thickness (mm)")
    plt.ylabel("Proportion of area (%)")
    plt.legend(loc='lower right')
    plt.grid(True, which='both', axis='both', alpha=0.5)
    return fig

def save_plots(fig_norm, fig_lin, labval):
    fig_norm.savefig(f"{labval}_normplot.png")
    fig_lin.savefig(f"{labval}_linplot.png")

def get_image_download_link(fig, labval, plot_type):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = buf.read()
    b64 = base64.b64encode(image_data).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{labval}_{plot_type}.png">Download {plot_type} plot</a>'

def get_descriptive_statistics(sorted_data):
    descriptive_stats = pd.Series(sorted_data).describe()
    return pd.DataFrame(descriptive_stats).transpose()

def main():
    st.set_page_config(layout="wide")
    st.title("Caisson Statistical Analysis")

    # Upload CSV
    st.header("Upload your CSV files")
    file1 = st.file_uploader("Upload first CSV", type=["csv"], key="file1")
    file2 = st.file_uploader("Upload second CSV", type=["csv"], key="file2")

    if file1 is not None:
        try:
            df1 = pd.read_csv(file1, header=None, index_col=False)
            st.write("First data loaded successfully!")
            st.write(df1.head())
            st.session_state.df1 = df1
            st.session_state.filename1 = file1.name
        except Exception as e:
            st.write("Error: Could not load the first file.")
            st.write(e)

    if file2 is not None:
        try:
            df2 = pd.read_csv(file2, header=None, index_col=False)
            st.write("Second data loaded successfully!")
            st.write(df2.head())
            st.session_state.df2 = df2
            st.session_state.filename2 = file2.name
        except Exception as e:
            st.write("Error: Could not load the second file.")
            st.write(e)

    if "df1" not in st.session_state or "df2" not in st.session_state:
        st.write("Please upload both CSV files first.")
    else:
        df1 = st.session_state.df1
        df2 = st.session_state.df2
        df1_numeric = df1.select_dtypes(include=[np.number])
        df2_numeric = df2.select_dtypes(include=[np.number])
        sorted_data1 = df1_numeric.stack().sort_values().values
        sorted_data2 = df2_numeric.stack().sort_values().values

        with st.form("plot_form"):
            with st.expander("Plot settings", expanded=True):
                labval1 = st.text_input("Enter a label for the first data:",
                                        value=st.session_state.filename1)
                labval2 = st.text_input("Enter a label for the second data:",
                                        value=st.session_state.filename2)

                if st.form_submit_button("Plot"):
                    if labval1 and labval2:
                        fig_norm = normplot(sorted_data1, sorted_data2, labval1, labval2)
                        fig_lin = linplot(sorted_data1, sorted_data2, labval1, labval2)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Normal Probability Plot")
                            st.pyplot(fig_norm)
                        with col2:
                            st.write("Scatter Plot")
                            st.pyplot(fig_lin)
                    else:
                        st.write("Please enter labels for the data.")

        # Download Plots
        if "fig_norm1" in locals() and "fig_lin1" in locals():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_image_download_link(fig_norm1, f"{labval1}_{labval2}", "normplot"),unsafe_allow_html=True)
            with col2:
                st.markdown(get_image_download_link(fig_lin1, f"{labval1}_{labval2}", "linplot"),unsafe_allow_html=True)
        # Add the descriptive statistics table
        st.write("Descriptive Statistics")
        col3, col4 = st.columns(2)
        with col3:
            st.write( f"{labval1}")
            descriptive_stats_table1 = get_descriptive_statistics(sorted_data1)
            st.dataframe(descriptive_stats_table1)
        with col4:
            st.write(f"{labval2}")
            descriptive_stats_table2 = get_descriptive_statistics(sorted_data2)
            st.dataframe(descriptive_stats_table2)

if __name__ == "__main__":
    main()
