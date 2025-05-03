def alloc_data(local_rank, num_workers, dataset_size):
    block_size = dataset_size // num_workers if dataset_size % num_workers == 0 else dataset_size // num_workers + 1
    local_start = local_rank * block_size
    if local_rank != num_workers - 1:
        local_end = local_start + block_size
    else:
        local_end = dataset_size
    return local_start, local_end

def test_alloc_data():
    print('Test alloc_data...')
    num_workers = 4
    dataset_size = 42
    for i in range(num_workers):
        print(alloc_data(i, num_workers, dataset_size))

if __name__ == "__main__":
    test_alloc_data()
