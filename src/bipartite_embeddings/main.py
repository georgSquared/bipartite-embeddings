from utils import load_graph, get_train_test_samples


def main():
    graph = load_graph()

    # Get train and test samples of the graph
    samples = get_train_test_samples(graph)


if __name__ == "__main__":
    main()
