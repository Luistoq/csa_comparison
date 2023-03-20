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

def normplot(data, labval, firstplot):
    mu = np.mean(data)
    sig = np.std(data)
    n = len(data)
    p = [(i - 0.5) / n for i in range(1, n + 1)]
    x = np.sort(data)
    y = norm.ppf(p)
    ticks_perc = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    ytv = np.percentile(y, ticks_perc)
    idxr = np.arange(1, n + 1)
    if n > 20000:
        idxr = np.concatenate((np.arange(1, 1000), np.arange(1001, 10000, 5), np.arange(10001, n, 50)))

    # Cap the maximum value of idxr to the length of x and y arrays
    idxr = np.minimum(idxr, len(x) - 1)

    fig, ax = plt.subplots()
    ylimval = (-4, 3)

    if np.max(y) < 3.1:
        ax.scatter(x[idxr], y[idxr], label=labval, s=2, linewidth=0,
                   ylim=ylimval)
    else:
        ax.scatter(x[idxr], y[idxr], label=labval, s=2, linewidth=0)
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

def linplot(data, labval, firstplot):
    fig, ax = plt.subplots()
    nd = len(data)
    pr = (2 * np.arange(1, nd + 1) - 1) / 2 / nd  # hazen plot plot pos for prob exceedance
    if firstplot == 1:
        plt1 = plt.scatter(np.sort(data), pr, s=2, linewidth=0,
                           label=labval, marker='o', color='darkorange')
    else:
        plt1 = plt.scatter(np.sort(data), pr, s=2, linewidth=0,
                           label=labval, marker='o', color='darkorange')
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
    st.header("Upload your CSV file")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file, header=None, index_col=False)
            st.write("Data loaded successfully!")
            st.write(df.head())
            st.session_state.df = df
            st.session_state.filename = file.name  # Store the filename in the session state
        except Exception as e:
            st.write("Error: Could not load the file.")
            st.write(e)

    if "df" not in st.session_state:
        st.write("Please upload a CSV file first.")
    else:
        df = st.session_state.df
        df_numeric = df.select_dtypes(include=[np.number])
        sorted_data = df_numeric.stack().sort_values().values

        with st.form("plot_form"):
            with st.expander("Plot settings", expanded=True):
                labval = st.text_input("Enter a label for the plots:",
                                       value=st.session_state.filename)  # Set the default value for labval

                if st.form_submit_button("Plot"):
                    if labval:
                        firstplot = 1
                        fig_norm = normplot(sorted_data, labval, firstplot)
                        fig_lin = linplot(sorted_data, labval, firstplot)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Normal Probability Plot")
                            st.pyplot(fig_norm)
                        with col2:
                            st.write("Scatter Plot")
                            st.pyplot(fig_lin)
                    else:
                        st.write("Please enter a label for the plots.")

        #Dowload Plots
        if "fig_norm" in locals() and "fig_lin" in locals():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_image_download_link(fig_norm, labval, "normplot"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_image_download_link(fig_lin, labval, "linplot"), unsafe_allow_html=True)

        # Add the descriptive statistics table
        st.write("Descriptive Statistics")
        descriptive_stats_table = get_descriptive_statistics(sorted_data)
        st.dataframe(descriptive_stats_table)

if __name__ == "__main__":
    main()
