package com.lucy;

public class CommentWithMetadata {
    public String id;
    public String text;
    public String publishedAt;
    public int likeCount;
    public String parentId; // null for top-level comments

    public CommentWithMetadata(String id, String text, String publishedAt, int likeCount, String parentId) {
        this.id = id;
        this.text = text;
        this.publishedAt = publishedAt;
        this.likeCount = likeCount;
        this.parentId = parentId;
    }
}