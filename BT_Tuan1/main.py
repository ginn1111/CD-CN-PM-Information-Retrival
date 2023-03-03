"""
    N19DCCN204: Phạm Văn Thuận
    N19DCNC126: Lê Hoài Nhân
    N19DCCN136: Nguyễn Trọng Tín
"""
import numpy as np
import re


def word_processing():
    # xử lý từ
    words = []
    with open("dataset/term-vocab", "r") as f:
        for line in f:
            # Tách các từ bằng khoảng trắng
            split_line = line.split()

            # Lọc bỏ các kí tự đặc biệt và số thứ tự và loại bỏ khoảng trắng ở đầu và cuối từ
            filtered_words = [word.strip("/").lower() for word in split_line if not word.isnumeric() and word.strip()]

            words.extend(filtered_words)

    for element in words:
        if element == "":
            words.remove(element)
    words = words[:-1]
    return words


def sentence_processing():
    sentences = []
    # xử lý đoạn văn
    with open("dataset/doc-text", "r") as f:
        data = f.read()

    # Tách các câu theo ký tự '/'
    sentences = data.split("/")

    # Loại bỏ các ký tự số trong mỗi câu
    for i, sentence in enumerate(sentences):
        words = []
        for word in sentence.split():
            if not word.isdigit():
                words.append(word)
        sentences[i] = " ".join(words).strip()

    sentences = sentences[:-1]

    # Tách từng từ trong câu
    sentences = [s.split(" ") for s in sentences]
    return sentences


def marking(words, sentences):
    # Tạo ma trận với các giá trị bằng 0
    matrix = np.zeros((len(words), len(sentences)))

    # Đánh dấu
    for r, w in enumerate(words):
        for c, s in enumerate(sentences):
            if w in s:
                matrix[r][c] = 1
    return matrix


def infix_to_postfix(query):
    # not có độ ưu tiên cao hơn
    priority = {"and": 1, "or": 1, "not": 2}
    stack = []
    postfix_list = []

    for token in query:
        if token not in ["and", "or", "not", "(", ")"]:
            postfix_list.append(token)
        elif token in priority.keys():
            while stack and stack[-1] != "(" and priority[token] <= priority.get(stack[-1], 0):
                postfix_list.append(stack.pop())
            stack.append(token)
        elif token == "(":
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                postfix_list.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
    while stack:
        postfix_list.append(stack.pop())
    return postfix_list


def marked_matrix():
    print("Processing...")
    words = word_processing()
    sentences = sentence_processing()
    s_size = len(sentences)

    matrix = marking(words, sentences)

    # In ma trận
    print("Processing marked matrix done")
    print(matrix)

    query = input("Query (example: A anb (B or C) and not D): ")
    # Tách các từ, dấu ngoặc, toán hạng truy vấn
    query = re.findall(r"\(|\)|AND|OR|NOT|\w+", query)

    # Nếu không tồn tại từ truy vấn trong bộ từ vựng
    for x in query:
        if x not in ["and", "or", "not"] and x not in words:
            print(x + " doesn't exist in vocabulary")
            return

    # Chuyển về dạng hậu tố
    postfix_l = infix_to_postfix(query)
    postfix_list = []
    for x in postfix_l:
        if x not in ["and", "or", "not"]:
            postfix_list.append(matrix[words.index(x)])
        else:
            postfix_list.append(x)

    # Tính toán bằng cách dịch ngược hậu tố
    stack = []
    operator = ["and", "or", "not"]
    for token in postfix_list:
        if token not in operator:
            stack.append(token)
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            if token == "and":
                res = []
                for i in range(s_size):
                    tmp = int(operand1[i]) & int(operand2[i])
                    res.append(int(tmp))
                stack.append(res)
            elif token == "or":
                res = []
                for i in range(s_size):
                    tmp = int(operand1[i]) | int(operand2[i])
                    res.append(int(tmp))
                stack.append(res)
            elif token == "not":
                stack.append(operand1)
                res = []
                for i in range(s_size):
                    tmp = ~int(operand2[i])
                    res.append(int(tmp))
                stack.append(res)
    print("Sentences are satisfied the query:")
    for id, val in enumerate(stack[0]):
        if val == 1:
            print("Sentence", id + 1)


def intersect(list_1, list_2):
    p1, p2 = 0, 0
    ans = []

    # Sử dụng FP
    # ans = list(set(filter(lambda x: x in list_2, list_1)))

    while p1 < len(list_1) and p2 < len(list_2):
        if list_1[p1] == list_2[p2]:
            ans.append(list_1[p1])
            p1 += 1
            p2 += 1
        elif list_1[p1] < list_2[p2]:
            p1 += 1
        else:
            p2 += 1
    return ans


def building_invered_index(words, sentences):
    my_dict = {}

    for id, s in enumerate(sentences):
        for w in s:
            if w in my_dict:
                my_dict[w].append(id + 1)
            else:
                my_dict[w] = []
                my_dict[w].append(id + 1)

    # Bỏ các phần tử trùng
    for key in my_dict:
        my_dict[key] = list(set(my_dict[key]))
        my_dict[key].sort()

    return my_dict


def inverted_index():
    print("Processing...")
    words = word_processing()
    sentences = sentence_processing()
    my_dict = building_invered_index(words, sentences)

    # Sort lại và hiển thị
    my_list = list(my_dict.items())
    my_list.sort(key=lambda x: x[0])

    print("Done")
    for x in my_list:
        print(x)

    # Lưu trữ chỉ mục ngược
    with open("inverted_intex.txt", "w") as f:
        # f.write(str(my_list))
        for x in my_list:
            f.write(str(x).replace("(", "").replace(")", ""))
            f.write("\n")

    # Truy vấn
    query = input("Query (A and B): ")
    # query = 'error and not function'

    # Tách các từ, dấu ngoặc, toán hạng truy vấn
    query = re.findall(r"\(|\)|AND|OR|NOT|\w+", query)

    print(query)

    # Nếu trong query có từ không nằm trong từ điển sẽ xảy ra ValueError
    # => Return luôn
    for x in query:
        if x not in ["and", "or", "not"] and x not in words:
            print(x + " not find in all sentences!")
            return

    list_1 = my_dict[query[0]]
    list_2 = my_dict[query[1]]

    sentence_id = intersect(list_1, list_2)
    print("Sentences that satisfied the query:")
    for x in sentence_id:
        print("Sentence", x)


def test():
    print("Processing...")
    words = word_processing()
    sentences = sentence_processing()
    my_dict = building_invered_index(words, sentences)

    # Sort lại và hiển thị
    my_list = list(my_dict.items())
    my_list.sort(key=lambda x: x[0])

    print("Done")
    for x in my_list:
        print(x)


def main():
    # test()
    marked_matrix()
    # inverted_index()


if __name__ == "__main__":
    main()
