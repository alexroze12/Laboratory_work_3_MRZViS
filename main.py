import numpy as np


def square_loss_function(expected_value, received_value):
    loss_value = pow((expected_value-received_value), 2)
    return loss_value


def derivative_square_loss_function(expected_value, received_value):
    loss_value = 2 * (expected_value-received_value)
    return loss_value


def convert_list_to_matrix(input_list):
    result_matrix = np.reshape(input_list, (1, -1))
    return result_matrix


def convert_data_to_list(data):
    return data.tolist()


def get_matrix_difference(first_matrix, second_matrix):
    first_matrix = np.matrix(first_matrix)
    second_matrix = np.matrix(second_matrix)
    return first_matrix-second_matrix


def get_multiplication_matrix_with_number(matrix, number):
    result = matrix.dot(number)
    return result


def get_matrix_multiplication(first_matrix, second_matrix):
    result = first_matrix.dot(second_matrix)
    return result


def activation_function_ELU(input_value, alpha):
    if input_value >= 0:
        return input_value
    else:
        return alpha * (np.exp(input_value)-1)


def derivative_activation_function_ELU(input_value, alpha):
    if input_value >= 0:
        return 1
    else:
        return alpha * (np.exp(input_value))


def initialization_of_network_layers(count_of_network_layers):
    network_layers = []
    network_layers.append(np.ones(count_of_network_layers[0] + count_of_network_layers[-1]))
    for index in range(1, len(count_of_network_layers)):
        network_layers.append(np.ones(count_of_network_layers[index]))
    return network_layers


def initialization_of_weight_matrix(count_of_network_layers, network_layers):
    network_weights = []
    for index in range(len(count_of_network_layers)-1):
        current_weight_matrix = np.random.randn(network_layers[index].size, network_layers[index+1].size) * np.sqrt(2 / network_layers[index].size)
        network_weights.append(current_weight_matrix)
    return network_weights


def get_fibonacci_values(number_of_generated_numbers):
    list_generated_fibonacci_values = list()
    for index in range(number_of_generated_numbers + 1):
        if index == 0:
            list_generated_fibonacci_values.append(1)
        elif index == 1:
            list_generated_fibonacci_values.append(1)
        else:
            sum_of_pred_values = list_generated_fibonacci_values[index-1] + list_generated_fibonacci_values[index-2]
            list_generated_fibonacci_values.append(sum_of_pred_values)
    return list_generated_fibonacci_values


def direct_error_distribution(network_layers, count_of_network_layers, x, weight):
    neuron_zeroing = 1
    for index in range(count_of_network_layers[0]):
        network_layers[0][index] = float(x[index])
    print(network_layers)
    if neuron_zeroing == 0:
        network_layers[0][count_of_network_layers[0]: -1] = np.zeros_like(network_layers[-1])
    else:
        network_layers[0][count_of_network_layers[0]: -1] = network_layers[-1]
    for index in range(1, len(count_of_network_layers) - 1):
        network_layers[index] = np.dot(network_layers[index - 1], weight[index - 1])
    if len(count_of_network_layers) - 2 >= 0:
        last_layer = len(count_of_network_layers) - 1
        network_layers[last_layer] = np.dot(network_layers[last_layer - 1], weight[last_layer - 1])
    return network_layers[-1]


def inverse_error_distribution(expected_value, network_layers, count_of_layers, network_weights, learning_step, momentum):
    errors = []
    current_delta_weight = []
    loss_error = square_loss_function(expected_value, network_layers[-1])
    loss_error_last_layer = derivative_square_loss_function(expected_value, network_layers[-1])
    errors.append(loss_error_last_layer)
    for index in range(len(count_of_layers) - 2, 0, -1):
        current_delta_weight = [0 for index in range(len(network_weights[index]))]
        current_error = np.dot(errors[0], np.dot(network_weights[index].T, 1))
        errors.insert(0, current_error)
    for index in range(len(network_weights)):
        network_layer = convert_list_to_matrix(network_layers[index])
        current_error = convert_list_to_matrix(errors[index])
        current_error_weight = np.dot(network_layer.T, current_error)
        network_weights[index] += learning_step * current_error_weight + learning_step * momentum * current_delta_weight[index]
        current_delta_weight[index] = current_error_weight
    return loss_error


def get_summary_error(error_on_each_iteration):
    return sum(error_on_each_iteration)


def get_average_error(error_on_each_iterations):
    return get_summary_error(error_on_each_iterations)/len(error_on_each_iterations)


def get_error_for_hidden_layer(weight_matrix_2, reference_value, output_value_neural_network):
    result = get_multiplication_matrix_with_number(get_matrix_difference(output_value_neural_network, reference_value), weight_matrix_2)
    return result


def calculate_weight_matrix_1(input_layer, old_weights, alpha, weight_matrix_2, reference_value, output_value_network):
    new_weights = get_matrix_multiplication(input_layer[:].transpose, old_weights)
    hidden_output = derivative_activation_function_ELU(new_weights, alpha)
    total_error = get_error_for_hidden_layer(weight_matrix_2, reference_value, output_value_network)


def get_data_tuple_for_sample(index, data, input_value, output_value):
    input_data_tuple = data[index: index + input_value]
    output_data_tuple = data[index + input_value: index + input_value + output_value]
    return [input_data_tuple, output_data_tuple]


def learn_model(count_of_network_layers, datasets, number_of_iterations, target_error, learning_step, momentum):
    print(datasets)
    network_layers = initialization_of_network_layers(neural_shape)
    weight = initialization_of_weight_matrix(count_of_network_layers, network_layers)
    summary_errors_on_each_iterations = []
    for i in range(number_of_iterations):
        errors_on_each_iterations = []
        for index in range(len(datasets)):
            result = direct_error_distribution(network_layers, count_of_network_layers, datasets[index][0], weight)
            error = inverse_error_distribution(datasets[index][1], network_layers, count_of_network_layers, weight, learning_step, momentum)
            errors_on_each_iterations.append(error)
        average_error = get_average_error(errors_on_each_iterations)
        if average_error <= target_error:
            break
        summary_errors_on_each_iterations.append(average_error)
    results_array = list()

    for index in range(len(datasets)):
        result = direct_error_distribution(network_layers, count_of_network_layers, datasets[index][0], weight)
        result = convert_data_to_list(result)
        result = [int(round(curr_result)) for curr_result in result]
        print(f'Input: {datasets[index][0]}, Received value: {result}. Expected result: {datasets[index][1]}')
        if datasets[1] == result:
            results_array.append(1)
        else:
            results_array.append(0)
    return summary_errors_on_each_iterations


if __name__ == "__main__":
    neural_shape = [2, 7, 1]
    number_of_generated_values = 8
    generated_values = get_fibonacci_values(number_of_generated_values)
    dataset_1 = []
    for i in range(len(generated_values)-2):
        dataset = get_data_tuple_for_sample(i, generated_values, neural_shape[0], neural_shape[2])
        dataset_1.append(dataset)
    print(dataset_1)
    b = learn_model(neural_shape, dataset_1, 1000, 0.03, 0.000003, 0.01)
