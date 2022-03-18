# 참고사이트: https://didu-story.tistory.com/102?category=952805
class CBOWClassifier(nn.Module): # Simplified cbow Model
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        """
        매개변수3개로 제어된다:
            vocabulary_size (int): 어휘 사전 크기, 임베딩 개수와 예측 벡터 크기를 결정합니다
            embedding_size (int): 임베딩 크기
            padding_idx (int): 기본값 0; 임베딩은 이 인덱스를 사용하지 않습니다
        """
        super(CBOWClassifier, self).__init__()

        # padding idx는 기본값이 0 이지만, 데이터 포인트의 길이가 모두 같지 않을 때
        # embedding층에 패딩하는데 사용되는 매개변수이다.
        self.embedding =  nn.Embedding(num_embeddings=vocabulary_size, 
                                       embedding_dim=embedding_size,
                                       padding_idx=padding_idx)
        # linear층에서 문맥벡터를 사용하여 예측벡터를 계산한다
        # 예측벡터는 어휘사전에 대한 확률분포
        self.fc1 = nn.Linear(in_features=embedding_size,
                             out_features=vocabulary_size)

    def forward(self, x_in, apply_softmax=False):
        """분류기의 정방향 계산
        
        매개변수:
            x_in (torch.Tensor): 입력 데이터 텐서 x_in.shape는 (batch, input_dim)입니다.
            apply_softmax (bool): 소프트맥스 활성화 함수를 위한 플래그
                                  크로스-엔트로피 손실을 사용하려면 False로 지정합니다
        반환값:
            결과 텐서. tensor.shape은 (batch, output_dim)입니다.
        """
        x_embedded_sum = F.dropout(self.embedding(x_in).sum(dim=1), 0.3)
        y_out = self.fc1(x_embedded_sum)
        
        ## 출력: 소프트맥스함수는 선택적으로 계산할 예정 / 수치적 계산 낭비 발생, 불안정 발생 
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        return y_out